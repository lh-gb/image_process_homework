"""命令行：单图、批量、实验、桌面界面。默认路径以项目目录为基准。"""
from pathlib import Path
import argparse
import csv
import json
import sys
from datetime import datetime
from docscan.config import DetectionConfig, EnhancementConfig
from docscan.io import load_image, save_json
from docscan.pipeline import process_document, export_result, ProcessingError

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

ROOT=Path(__file__).resolve().parent


def parser():
    p=argparse.ArgumentParser(description='手机拍摄文档透视校正与增强 / OpenCV')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('process','batch'):
        q=sub.add_parser(name,help='处理一张图片' if name=='process' else '批量处理文件夹')
        q.add_argument('input',type=Path)
        q.add_argument('--output',type=Path,default=None)
        q.add_argument('--detector',choices=['baseline','robust'],default='robust')
        q.add_argument('--enhancement',choices=['otsu','adaptive','illumination'],default='illumination')
        q.add_argument('--max-dimension',type=int,default=1280)
        q.add_argument('--clahe',type=float,default=0.0,help='可选CLAHE clip limit，默认关闭')
        if name=='process':
            q.add_argument('--corners',type=Path,help='手动角点JSON: [[x,y],...] 或含 corners 的对象')
        else:
            q.add_argument('--pattern',default='doc_*.jpg',help='默认只处理新增50张，避免把课件截图混入实验')
    q=sub.add_parser('evaluate',help='有四角标注数据的基线、消融与增强对比')
    q.add_argument('--dataset',type=Path,default=ROOT/'dataset')
    q.add_argument('--output',type=Path,default=ROOT/'output'/'experiment')
    q.add_argument('--split',choices=['all','development','test'],default='all')
    q.add_argument('--iou-threshold',type=float,default=.8)
    sub.add_parser('gui',help='打开桌面演示界面')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command=='gui':
        from app import main as gui_main
        gui_main(); return 0
    if args.command=='evaluate':
        from docscan.evaluation import evaluate_dataset
        summary=evaluate_dataset(args.dataset,args.output,split=args.split,iou_threshold=args.iou_threshold)
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        print(f'实验结果：{args.output.resolve()}')
        return 0
    dc=DetectionConfig(mode=args.detector,max_dimension=args.max_dimension)
    ec=EnhancementConfig(mode=args.enhancement,clahe_clip=args.clahe)
    if args.command=='process':
        corners=None
        if args.corners:
            obj=json.loads(args.corners.read_text('utf-8-sig'))
            if isinstance(obj,dict):
                if 'corners' in obj:
                    corners=obj['corners']
                elif 'corners_xy' in obj:
                    corners=obj['corners_xy']
                else:
                    raise ValueError('角点 JSON 对象必须包含 corners 或 corners_xy')
            else:
                corners=obj
        result=process_document(load_image(args.input),dc,ec,corners=corners)
        out=args.output or ROOT/'output'/'single'/args.input.stem
        paths=export_result(result,out)
        print(json.dumps({'output':str(out.resolve()),'files':paths,'timing_ms':result.metadata['timing_ms']},ensure_ascii=False,indent=2))
        return 0
    if not args.input.is_dir(): raise ValueError(f'文件夹不存在：{args.input}')
    files=sorted(p for p in args.input.glob(args.pattern) if p.is_file())
    if not files: raise ValueError('没有匹配图片，请检查路径与 --pattern')
    out=args.output or ROOT/'output'/'batch'
    out.mkdir(parents=True,exist_ok=True)
    results=[]
    for i,path in enumerate(files,1):
        try:
            result=process_document(load_image(path),dc,ec)
            export_result(result,out/path.stem)
            row={'filename':path.name,'status':'ok','total_ms':result.metadata['timing_ms']['total'],'error':''}
        except (ProcessingError,ValueError,OSError) as exc:
            row={'filename':path.name,'status':'failed','total_ms':'','error':str(exc)}
            save_json(out/path.stem/'error.json',row)
        results.append(row)
        print(f"[{i}/{len(files)}] {path.name}: {row['status']}",flush=True)
    with (out/'batch_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    n=sum(r['status']=='ok' for r in results)
    save_json(out/'batch_summary.json',{'total':len(results),'processed':n,'failed':len(results)-n,
                                     'note':'processed仅表示产生候选与输出，不代表几何正确；准确率需标注评估。'})
    print(f'完成 {n}/{len(results)}，失败 {len(results)-n}，输出 {out.resolve()}')
    return 0 if n else 2

if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ProcessingError,ValueError,OSError) as exc:
        print(f'处理失败：{exc}',file=sys.stderr)
        raise SystemExit(2)
