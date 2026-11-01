"""端到端扫描。失败显式报告，手动四角是用户输入，不读取数据集真值。"""
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from .config import DetectionConfig, EnhancementConfig
from .detection import detect_document
from .geometry import order_points, rectify_document
from .enhancement import enhance_document
from .io import save_image, save_json, make_grid

class ProcessingError(RuntimeError):
    pass

@dataclass
class ScanResult:
    original: np.ndarray
    detected: np.ndarray
    rectified: np.ndarray
    enhanced: np.ndarray
    binary: np.ndarray
    steps: dict
    metadata: dict
    corners: np.ndarray


def process_document(image_bgr, detection_config=None, enhancement_config=None, corners=None):
    dc,ec=detection_config or DetectionConfig(),enhancement_config or EnhancementConfig()
    start=perf_counter()
    image=np.asarray(image_bgr)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ProcessingError('输入必须是 uint8 BGR 彩色图片')
    detection=None
    if corners is None:
        detection=detect_document(image,dc)
        if detection.corners is None:
            raise ProcessingError('未找到可靠四边形。请尝试调整拍摄背景，或在界面中手动标记四角。')
        points=order_points(detection.corners)
    else:
        try: points=order_points(corners)
        except ValueError as exc: raise ProcessingError(str(exc)) from exc
    h,w=image.shape[:2]
    if not all(0<=x<w and 0<=y<h for x,y in points):
        raise ProcessingError('角点位于图像外部')
    t_detect=perf_counter()
    try: rectified=rectify_document(image,points)
    except ValueError as exc: raise ProcessingError(str(exc)) from exc
    t_rectify=perf_counter()
    stages=enhance_document(rectified,ec)
    t_end=perf_counter()
    overlay=image.copy()
    thickness=max(2,round(max(h,w)/700))
    cv2.polylines(overlay,[np.rint(points).astype(np.int32)],True,(40,210,80),thickness,cv2.LINE_AA)
    for i,(x,y) in enumerate(points):
        cv2.circle(overlay,(round(float(x)),round(float(y))),thickness*3,(30,80,240),-1)
        cv2.putText(overlay,str(i+1),(round(float(x))+8,round(float(y))-8),
                    cv2.FONT_HERSHEY_SIMPLEX,.7,(30,80,240),2,cv2.LINE_AA)
    metadata={'corner_source':'manual' if corners is not None else 'automatic',
              'corners_xy':points.tolist(),'corner_order':'cyclic, image-space upper-left start',
              'input_size':[w,h],'rectified_size':[rectified.shape[1],rectified.shape[0]],
              'aspect_assumption':'opposite-edge-length estimate; no known physical paper ratio',
              'detection_config':asdict(dc),'enhancement_config':asdict(ec),
              'detection_score':float(detection.score) if detection else None,
              'candidate_count':detection.candidate_count if detection else 0,
              'diagnostics':detection.diagnostics if detection else {},
              'timing_ms':{'detection':(t_detect-start)*1000,'rectification':(t_rectify-t_detect)*1000,
                           'enhancement':(t_end-t_rectify)*1000,'total':(t_end-start)*1000}}
    return ScanResult(image.copy(),overlay,rectified,stages['enhanced'],stages['binary'],stages,metadata,points)


def export_result(result,output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    images={'original':result.original,'detected':result.detected,'rectified':result.rectified,
            'enhanced':result.enhanced,'binary':result.binary}
    images.update({k:v for k,v in result.steps.items() if k not in images})
    paths={k:save_image(out/f'{k}.png',v) for k,v in images.items()}
    paths['comparison']=save_image(out/'comparison.jpg',make_grid(
        [result.original,result.detected,result.rectified,result.steps['gray'],result.enhanced,result.binary],
        ['Original','Detected corners','Perspective rectified','Grayscale','Enhanced','Binarized']))
    save_json(out/'result.json',result.metadata)
    paths['metadata']=str(out/'result.json')
    return paths
