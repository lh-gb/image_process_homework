"""可复现的文档定位与增强对比实验。

原图坐标系 IoU 为本项目自定义指标，不等同于 SmartDoc 官方的文档坐标系指标。
没有干净扫描参考或像素级文字标注时，仅报告增强的描述性统计与可视化。
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .config import DetectionConfig, EnhancementConfig
from .detection import detect_document
from .enhancement import enhance_document
from .geometry import corner_error, polygon_iou, rectify_document
from .io import load_image, save_image


def _write_csv(path, rows, fieldnames=None):
    """UTF-8 BOM 便于中文 Windows Excel 直接打开。"""
    fieldnames = fieldnames or (list(rows[0]) if rows else [])
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_annotations(dataset_dir):
    primary = dataset_dir / '_metadata' / 'annotations.json'
    fallback = Path(__file__).resolve().parents[1] / 'resources' / 'smartdoc50_annotations.json'
    source = primary if primary.is_file() else fallback
    if not source.is_file():
        raise FileNotFoundError(
            '未找到真实四角标注。需要 dataset/_metadata/annotations.json，'
            '或 code/resources/smartdoc50_annotations.json；不能从算法结果生成评价真值。'
        )
    data = json.loads(source.read_text(encoding='utf-8-sig'))
    images = data.get('images', [])
    if not images:
        raise ValueError(f'标注中没有 images 记录：{source}')
    seen, doc_types = set(), {}
    for item in images:
        required = {'filename', 'document_id', 'document_type', 'background', 'corners'}
        if not required.issubset(item):
            raise ValueError(f'标注缺少字段：{required - set(item)}')
        rel = Path(item['filename'])
        path = (dataset_dir / rel).resolve()
        if rel.is_absolute() or dataset_dir.resolve() not in path.parents:
            raise ValueError(f'图片路径超出 dataset：{rel}')
        if item['filename'] in seen:
            raise ValueError(f'重复的标注图片名：{item["filename"]}')
        seen.add(item['filename'])
        if not path.is_file():
            raise FileNotFoundError(f'标注对应的图片不存在：{path}')
        points = np.asarray(item['corners'], dtype=np.float32)
        if not _valid_quad(points):
            raise ValueError(f'四角标注不是有效凸四边形：{rel}')
        doc_id, doc_type = item['document_id'], item['document_type']
        if doc_id in doc_types and doc_types[doc_id] != doc_type:
            raise ValueError(f'同一 document_id 对应不同 document_type：{doc_id}')
        doc_types[doc_id] = doc_type
        expected = item.get('sha256')
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'图片与来源标注的 SHA-256 不一致：{path}')
    return sorted(images, key=lambda item: item['filename']), source


def _valid_quad(points):
    if points is None:
        return False
    p = np.asarray(points, dtype=np.float32)
    return bool(p.shape == (4, 2) and np.isfinite(p).all()
                and abs(cv2.contourArea(p)) > 1.0 and cv2.isContourConvex(p))


def _assign_splits(images):
    """按类别、文档 ID 固定分层；跨背景的同一文档不跨组。"""
    docs_by_type = defaultdict(set)
    for item in images:
        docs_by_type[item['document_type']].add(item['document_id'])
    assignment = {}
    for doc_type, documents in sorted(docs_by_type.items()):
        documents = sorted(documents)
        if len(documents) < 3:
            raise ValueError(f'{doc_type} 仅有 {len(documents)} 个文档，无法按前 2 个 development、其余 test 划分')
        for index, doc_id in enumerate(documents):
            assignment[doc_id] = 'development' if index < 2 else 'test'
    return assignment


def _group_summary(rows, threshold):
    n = len(rows)
    if not n:
        return {'n': 0}
    detected = [r for r in rows if r['detected']]
    corner_values = [r['corner_error_normalized'] for r in detected
                     if r['corner_error_normalized'] is not None]
    times = [r['detection_ms'] for r in rows]
    return {
        'n': n,
        'unique_documents': len({r['document_id'] for r in rows}),
        'detected_count': len(detected),
        'not_detected_count': n - len(detected),
        'detection_rate': len(detected) / n,
        'mean_iou': float(np.mean([r['iou'] for r in rows])),
        'median_iou': float(np.median([r['iou'] for r in rows])),
        'success_at_threshold': sum(r['iou'] >= threshold for r in rows) / n,
        'success_at_0_9': sum(r['iou'] >= 0.9 for r in rows) / n,
        'detected_but_below_threshold': sum(r['iou'] < threshold for r in detected),
        'mean_corner_error_detected': float(np.mean(corner_values)) if corner_values else None,
        'mean_detection_ms': float(np.mean(times)),
        'median_detection_ms': float(np.median(times)),
    }


def _paired_bootstrap(rows, iterations=1000, seed=42):
    """按文档聚类的配对 bootstrap；估计量为等权文档平均 IoU 差。"""
    paired = defaultdict(dict)
    for row in rows:
        if row['method'] in ('baseline', 'robust'):
            paired[row['filename']][row['method']] = row
    deltas = defaultdict(list)
    for pair in paired.values():
        if 'baseline' in pair and 'robust' in pair:
            base, improved = pair['baseline'], pair['robust']
            deltas[base['document_id']].append(improved['iou'] - base['iou'])
    means = np.array([np.mean(v) for _, v in sorted(deltas.items())], dtype=np.float64)
    if not len(means):
        return {'n_documents': 0, 'iterations': iterations, 'seed': seed}
    rng = np.random.default_rng(seed)
    draws = rng.choice(means, size=(iterations, len(means)), replace=True).mean(axis=1)
    return {
        'estimand': 'document_macro_mean_iou_robust_minus_baseline',
        'n_documents': len(means),
        'point_estimate': float(means.mean()),
        'percentile_ci95': [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        'iterations': iterations, 'seed': seed,
        'note': '文档为重采样单元，文档内所有背景共同保留；非随机小子集的区间不能代表完整 SmartDoc 总体。',
    }


def _descriptive_enhancement(gray, binary):
    """仅描述输出分布，不将白底均匀程度或前景比例解释为图像质量。"""
    grid_y = np.linspace(0, gray.shape[0], 9, dtype=int)
    grid_x = np.linspace(0, gray.shape[1], 9, dtype=int)
    tile_p90 = []
    for yi in range(8):
        for xi in range(8):
            tile = gray[grid_y[yi]:grid_y[yi+1], grid_x[xi]:grid_x[xi+1]]
            if tile.size:
                tile_p90.append(float(np.percentile(tile, 90)))
    return {'mean_gray': float(np.mean(gray)),
            'tile_p90_std': float(np.std(tile_p90)) if tile_p90 else None,
            'black_pixel_fraction': float(np.mean(binary < 128))}


def _thumbnail(bgr, size=(320, 180)):
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    result = Image.new('RGB', size, '#f2f4f7')
    thumb = ImageOps.contain(im, size)
    result.paste(thumb, ((size[0]-thumb.width)//2, (size[1]-thumb.height)//2))
    return result


def _overlay(image, truth, predictions):
    canvas = image.copy()
    width = max(2, round(max(image.shape[:2]) / 600))
    cv2.polylines(canvas, [np.rint(truth).astype(np.int32)], True, (30, 195, 40), width)
    for name, color in [('baseline', (20, 135, 250)), ('robust', (240, 90, 25))]:
        points = predictions.get(name)
        if _valid_quad(points):
            cv2.polylines(canvas, [np.rint(points).astype(np.int32)], True, color, width)
    return canvas


def _contact_sheet(tiles, path, columns=5, tile_size=(336, 222), title=''):
    if not tiles:
        return
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new('RGB', (columns*tile_size[0], rows*tile_size[1]+32), '#ffffff')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), title, fill='#1c2434')
    for i, (image, line1, line2) in enumerate(tiles):
        x, y = (i % columns)*tile_size[0], (i // columns)*tile_size[1]+32
        sheet.paste(image, (x+8, y+4))
        draw.text((x+8, y+188), line1, fill='#1c2434')
        draw.text((x+8, y+204), line2, fill='#465166')
    sheet.save(path, quality=92)


def _enhancement_panel(rectified, results, filename):
    size = (240, 300)
    panel = Image.new('RGB', (4*size[0], size[1]+44), '#ffffff')
    draw = ImageDraw.Draw(panel)
    draw.text((8, 6), filename + ' | same predicted geometry', fill='black')
    items = [('rectified', rectified)] + [(m, results[m]['binary']) for m in ('otsu', 'adaptive', 'illumination')]
    for i, (label, arr) in enumerate(items):
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        panel.paste(_thumbnail(arr, size), (i*size[0], 26))
        draw.text((i*size[0]+8, size[1]+30), label, fill='black')
    return panel


def _make_plots(summary, output_dir, threshold):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return {'generated': False, 'reason': 'matplotlib not installed; CSV and contact sheets remain available'}
    methods = list(summary['detection'])
    labels = ['baseline', 'robust', 'no norm.', 'no regions', 'area score']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), constrained_layout=True)
    scores = [summary['detection'][m]['all'] for m in methods]
    series = [([s['mean_iou'] for s in scores], 'Mean image-plane IoU', (0, 1)),
              ([s['success_at_threshold'] for s in scores], f'Success rate (IoU >= {threshold:g})', (0, 1)),
              ([s['mean_detection_ms'] for s in scores], 'Mean detection time (ms)', None)]
    for ax, (values, title, ylim) in zip(axes, series):
        bars = ax.bar(labels, values, color=['#94a3b8', '#2563eb', '#60a5fa', '#60a5fa', '#60a5fa'])
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=25)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        if ylim:
            ax.set_ylim(*ylim)
        for bar, value in zip(bars, values):
            ax.annotate(f'{value:.3f}' if ylim else f'{value:.1f}',
                        (bar.get_x()+bar.get_width()/2, value), xytext=(0, 3),
                        textcoords='offset points', ha='center', fontsize=8)
    fig.savefig(output_dir/'detection_comparison.png', dpi=160)
    plt.close(fig)
    return {'generated': True, 'file': 'detection_comparison.png'}


def _markdown(summary, threshold):
    lines = ['# 文档透视校正与增强：可复现实验记录', '',
             '本文件由真实运行结果生成，不预先指定哪一种方法更好。', '',
             '## 数据与划分', '',
             f'- 本次评价图片：{summary["image_count"]} 张；所选分组：`{summary["requested_split"]}`。',
             '- 按 document_type 分层，每类 document_id 排序后前 2 个为 development，其余为 test（原始 30 文档为 12/18）。',
             '- 同一文档的不同背景图像必须处于同一分组。全体样本的分组见 `split.csv`。',
             '- development 用于调试和参数选择，test 应在方法固定后一次性评价；程序不把任何真值传入检测或增强。',
             '- 此 50 张公开数据子集是有选择的样本，非完整 SmartDoc 官方评测；原始帧筛选面积下限为 8%。', '',
             '## 方法与消融', '',
             '- `baseline`：基础边缘/轮廓方案。',
             '- `robust`：完整候选生成和组合评分方案。',
             '- `robust_no_normalization`：关闭归一化边缘提议。',
             '- `robust_no_regions`：关闭阈值区域提议。',
             '- `robust_area`：保留改进候选，仅按面积评分。',
             '- 检测 score 仅为候选排序分数，不是经过校准的置信概率。',
             '- 三种增强 `otsu / adaptive / illumination` 共用相同 robust 预测裁剪，失败时不以真实四角代替预测。', '',
             '## 指标定义', '',
             '- IoU：原图像素坐标系下预测凸四边形与标注四边形的交并比。未检测或无效四边形按 0 计入均值。',
             f'- 成功率：IoU ≥ {threshold:g}；同时报告 IoU ≥ 0.9 的成功率。',
             '- 检出率只表示返回有效四边形，不等于定位正确率。',
             '- 四角误差：凸包统一绕序后四种循环起点配对的最小平均欧氏误差，除以原图对角线。仅对有效检测结果统计，必须结合失败数量阅读。',
             '- 本项目原图 IoU 与 SmartDoc 官方将区域投影到文档坐标系的协议不同，不应直接比较论文中的官方数值。',
             '- 每个配置在首图预热一次，预热不计时；报告实际 CPU 检测与四边形有效性检查耗时，不含读盘、增强和可视化。耗时受硬件与线程影响。', '',
             '## 定位结果', '',
             '| 方法 | 分组 | n | 平均 IoU | 成功率 | @0.9 | 检出率 | 四角误差（归一化） | 耗时 ms |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for method, groups in summary['detection'].items():
        for group in ('all', 'development', 'test'):
            s = groups[group]
            if not s['n']:
                continue
            err = s['mean_corner_error_detected']
            err_text = f'{err:.4f}' if err is not None else '—'
            lines.append(f'| {method} | {group} | {s["n"]} | {s["mean_iou"]:.4f} | {s["success_at_threshold"]:.1%} | {s["success_at_0_9"]:.1%} | {s["detection_rate"]:.1%} | {err_text} | {s["mean_detection_ms"]:.1f} |')
    lines += ['', '## 不确定性', '',
              '对 robust−baseline 的 IoU 差按 document_id 做配对聚类 bootstrap（固定随机种子 42，1,000 次）。',
              '估计量为每文档先平均、再对文档等权平均的差值，因此与逐图平均 IoU 差可能略有不同。', '']
    for group, b in summary['paired_document_bootstrap'].items():
        if b.get('n_documents'):
            lo, hi = b['percentile_ci95']
            lines.append(f'- {group}：{b["n_documents"]} 个文档，差值 {b["point_estimate"]:.4f}，95% percentile CI [{lo:.4f}, {hi:.4f}]。')
    lines += ['', '## 增强结果的解释边界', '',
              '- 当前数据没有配对干净扫描参考，也没有像素级文字真值。因此不计算 PSNR/SSIM，不声称输出统计证明识别率或视觉质量提高。',
              '- `mean_gray` 为输出灰度均值；`tile_p90_std` 为 8×8 网格各块灰度 90 分位数的标准差，仅作为白底均匀性的描述性代理；`black_pixel_fraction` 为二值图黑像素比例。',
              '- 整张空白图也可能具有极低背景波动，因此不能把这些代理量作为质量分数或优化目标。应结合文字局部放大、断笔/粘连和阴影残留分析。',
              '- 增强耗时不含透视矫正；每张图的预测裁剪相同。检测失败的图没有增强记录，不能据此把增强结果解释成全数据集端到端成功率。',
              '- 当前 demo 不含 OCR。若增加 OCR 评价，应使用独立人工文字转录，固定识别器并报告 CER/WER。', '',
              '## 输出', '',
              '- `detection_results.csv`：逐图、逐检测配置的指标和错误。',
              '- `enhancement_results.csv`：固定预测几何下的三种增强描述统计。',
              '- `summary.json`：分组汇总、具体配置、环境信息及 bootstrap。',
              '- `detection_contact_sheet.jpg`：全样本定位总览；绿色=真值，橙色=baseline，蓝色=robust。',
              '- `cases/`：有效结果的较高/较低 IoU 案例，选择仅用于事后诊断，不用于改变计分。',
              '- `enhancement_contact_sheet.jpg`：每背景最多两张的增强对比，选取规则不依赖方法优劣。',
              '- `detection_comparison.png`：若 matplotlib 可用则生成。', '',
              '本记录中的局限与失败案例应进入课程汇报，而非仅展示处理成功的样例。', '']
    return '\n'.join(lines)


def evaluate_dataset(dataset_dir, output_dir, *, split='all', iou_threshold=0.8):
    """运行检测消融和固定预测几何上的增强对比，返回可 JSON 序列化的汇总。"""
    if split == 'dev':
        split = 'development'
    if split not in ('all', 'development', 'test'):
        raise ValueError('split 只能为 all、development（或 dev）、test')
    if not 0 < iou_threshold <= 1:
        raise ValueError('iou_threshold 必须在 (0, 1] 内')
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    if output_dir.resolve() == dataset_dir.resolve() or dataset_dir.resolve() in output_dir.resolve().parents:
        raise ValueError('实验输出应放到 code/output 等目录，不得写入原始 dataset 内')
    images, annotation_source = _load_annotations(dataset_dir)
    assignment = _assign_splits(images)
    split_rows = [{**{k: item[k] for k in ('filename','document_id','document_type','background')},
                   'split': assignment[item['document_id']]} for item in images]
    selected = [item for item in images if split == 'all' or assignment[item['document_id']] == split]
    if not selected:
        raise ValueError(f'分组 {split} 没有可评价图片')
    configs = {
        'baseline': DetectionConfig(mode='baseline'),
        'robust': DetectionConfig(mode='robust'),
        'robust_no_normalization': DetectionConfig(mode='robust', use_normalized_edges=False),
        'robust_no_regions': DetectionConfig(mode='robust', use_threshold_proposals=False),
        'robust_area': DetectionConfig(mode='robust', scoring='area'),
    }
    enhancement_configs = {name: EnhancementConfig(mode=name) for name in ('otsu','adaptive','illumination')}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'cases').mkdir(exist_ok=True)
    _write_csv(output_dir/'split.csv', split_rows)
    first_image = load_image(dataset_dir/selected[0]['filename'])
    for cfg in configs.values():
        detect_document(first_image, cfg)
    detection_rows, enhancement_rows, detection_tiles = [], [], []
    predictions_by_file, robust_by_file = {}, {}
    enhancement_panels, preview_per_background = [], defaultdict(int)
    started = time.perf_counter()
    for index, item in enumerate(selected, 1):
        image = load_image(dataset_dir/item['filename'])
        h, w = image.shape[:2]
        if (item.get('width') and int(item['width']) != w) or (item.get('height') and int(item['height']) != h):
            raise ValueError(f'图片尺寸与标注不一致：{item["filename"]}')
        truth = np.asarray(item['corners'], dtype=np.float32)
        if not np.all((truth[:,0] >= 0) & (truth[:,0] < w) & (truth[:,1] >= 0) & (truth[:,1] < h)):
            raise ValueError(f'标注角点超出图片：{item["filename"]}')
        common = {key: item[key] for key in ('filename','document_id','document_type','background')}
        common['split'] = assignment[item['document_id']]
        predictions, rows_for_image = {}, {}
        for name, cfg in configs.items():
            t0 = time.perf_counter()
            error, result, points = '', None, None
            try:
                result = detect_document(image, cfg)
                points = result.corners
                if not _valid_quad(points):
                    points = None
                    error = 'no_valid_document_quad'
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            elapsed = (time.perf_counter()-t0)*1000
            valid = points is not None
            iou = float(polygon_iou(points, truth)) if valid else 0.0
            err = float(corner_error(points, truth, image.shape)) if valid else None
            row = {**common, 'method': name, 'detected': int(valid),
                   'iou': iou, 'corner_error_normalized': err,
                   'detection_ms': elapsed,
                   'score': float(result.score) if result is not None else None,
                   'candidate_count': int(result.candidate_count) if result is not None else 0,
                   'error': error}
            detection_rows.append(row)
            rows_for_image[name] = row
            predictions[name] = np.asarray(points).copy() if valid else None
        predictions_by_file[item['filename']] = predictions
        robust_by_file[item['filename']] = rows_for_image['robust']
        overlay = _overlay(image, truth, predictions)
        detection_tiles.append((_thumbnail(overlay),
            f'{item["filename"]} | {common["split"]}',
            f'IoU base={rows_for_image["baseline"]["iou"]:.2f} robust={rows_for_image["robust"]["iou"]:.2f}'))
        if predictions['robust'] is not None:
            t0 = time.perf_counter()
            rectified = rectify_document(image, predictions['robust'])
            rectify_ms = (time.perf_counter()-t0)*1000
            enhanced = {}
            for mode, cfg in enhancement_configs.items():
                t0 = time.perf_counter()
                outputs = enhance_document(rectified, cfg)
                elapsed = (time.perf_counter()-t0)*1000
                enhanced[mode] = outputs
                stats = _descriptive_enhancement(outputs['enhanced'], outputs['binary'])
                enhancement_rows.append({**common, 'mode': mode,
                    'geometry_source': 'robust_prediction', 'geometry_iou': rows_for_image['robust']['iou'],
                    'rectified_width': rectified.shape[1], 'rectified_height': rectified.shape[0],
                    'rectification_ms': rectify_ms, 'enhancement_ms': elapsed, **stats})
            if preview_per_background[item['background']] < 2:
                enhancement_panels.append(_enhancement_panel(rectified, enhanced, item['filename']))
                preview_per_background[item['background']] += 1
        print(f'[{index}/{len(selected)}] {item["filename"]}: robust IoU={rows_for_image["robust"]["iou"]:.3f}', flush=True)
    summary = {
        'requested_split': split, 'image_count': len(selected),
        'annotation_source': (annotation_source.resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
                              if annotation_source.resolve().is_relative_to(Path(__file__).resolve().parents[1])
                              else str(annotation_source)),
        'success_iou_threshold': iou_threshold,
        'iou_protocol': 'Original image-plane convex polygon IoU; not official SmartDoc document-plane IoU',
        'corner_error_protocol': 'Minimum cyclic point pairing after canonical convex ordering / original image diagonal',
        'split_protocol': 'Per document_type, sorted first 2 document_id development, all remaining test; group by document_id',
        'warmup': 'One detection call per configuration on the first selected image, excluded from timing',
        'environment': {'python': sys.version.split()[0], 'opencv': cv2.__version__,
                        'numpy': np.__version__, 'platform': platform.platform(),
                        'opencv_threads': cv2.getNumThreads()},
        'detection_configs': {name: asdict(cfg) for name,cfg in configs.items()},
        'enhancement_configs': {name: asdict(cfg) for name,cfg in enhancement_configs.items()},
        'detection': {}, 'enhancement_descriptive': {}, 'paired_document_bootstrap': {},
        'enhancement_warning': 'No clean scan or text-mask reference: descriptive output statistics are not image quality scores. No PSNR/SSIM/OCR accuracy claims.',
    }
    for name in configs:
        rows = [r for r in detection_rows if r['method'] == name]
        groups = {g: _group_summary(rows if g == 'all' else [r for r in rows if r['split'] == g], iou_threshold)
                  for g in ('all','development','test')}
        for key in ('background','document_type'):
            groups['by_'+key] = {value: _group_summary([r for r in rows if r[key] == value], iou_threshold)
                                   for value in sorted({r[key] for r in rows})}
        summary['detection'][name] = groups
    for group in ('all','development','test'):
        summary['paired_document_bootstrap'][group] = _paired_bootstrap(
            detection_rows if group == 'all' else [r for r in detection_rows if r['split'] == group])
    for mode in enhancement_configs:
        rows = [r for r in enhancement_rows if r['mode'] == mode]
        summary['enhancement_descriptive'][mode] = {'n': len(rows), **{
            'mean_'+k: float(np.mean([r[k] for r in rows])) if rows else None
            for k in ('enhancement_ms','mean_gray','tile_p90_std','black_pixel_fraction')}}
    _write_csv(output_dir/'detection_results.csv', detection_rows)
    _write_csv(output_dir/'enhancement_results.csv', enhancement_rows,
        fieldnames=list(enhancement_rows[0]) if enhancement_rows else ['filename','mode','geometry_source','geometry_iou','enhancement_ms'])
    _contact_sheet(detection_tiles, output_dir/'detection_contact_sheet.jpg',
        title='Green: ground truth | Orange: baseline | Blue: robust | original image-plane IoU')
    if enhancement_panels:
        panel = Image.new('RGB', (960, sum(p.height for p in enhancement_panels)), '#ffffff')
        y = 0
        for p in enhancement_panels:
            panel.paste(p, (0, y)); y += p.height
        panel.save(output_dir/'enhancement_contact_sheet.jpg', quality=92)
    ranked = sorted(selected, key=lambda item: robust_by_file[item['filename']]['iou'])
    case_items = [('lowest', x) for x in ranked[:3]] + [('highest', x) for x in ranked[-3:]]
    for label, item in case_items:
        image = load_image(dataset_dir/item['filename'])
        overlay = _overlay(image, np.asarray(item['corners'], dtype=np.float32), predictions_by_file[item['filename']])
        save_image(output_dir/'cases'/f'{label}_{Path(item["filename"]).stem}.jpg', overlay)
    try:
        summary['plots'] = _make_plots(summary, output_dir, iou_threshold)
    except Exception as exc:
        summary['plots'] = {'generated': False, 'reason': f'{type(exc).__name__}: {exc}'}
    summary['total_experiment_seconds'] = time.perf_counter()-started
    (output_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n','utf-8')
    (output_dir/'EXPERIMENT_REPORT.md').write_text(_markdown(summary, iou_threshold), 'utf-8')
    return summary
