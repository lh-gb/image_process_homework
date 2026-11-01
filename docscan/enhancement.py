"""文档增强：共享去噪下比较 Otsu、局部阈值和背景光照补偿。"""
from dataclasses import asdict
import cv2
import numpy as np
from .config import EnhancementConfig


def _odd_window(value, limit):
    limit = int(limit) if int(limit)%2 else int(limit)-1
    value = max(3,int(round(value)))
    value += 1-value%2
    return min(value,max(3,limit))


def enhance_document(image_bgr, config=None):
    cfg=config or EnhancementConfig()
    arr=np.asarray(image_bgr)
    if arr.dtype!=np.uint8 or arr.ndim not in (2,3) or min(arr.shape[:2])<3:
        raise ValueError('增强输入必须是至少3×3的uint8图像')
    if arr.ndim==3 and arr.shape[2]!=3:
        raise ValueError('彩色输入必须使用BGR三通道')
    gray=cv2.cvtColor(arr,cv2.COLOR_BGR2GRAY) if arr.ndim==3 else arr.copy()
    # 三种模式使用相同预处理，避免将不同去噪器混入增强对照。
    work=cv2.bilateralFilter(gray,5,25,3) if cfg.denoise else gray.copy()
    background=np.full_like(work,255)
    normalized=work.copy()
    if cfg.mode=='illumination':
        k=_odd_window(cfg.background_kernel or max(15,min(work.shape)/15),min(work.shape))
        # 闭运算填补比结构元小的暗文字，再平滑得到缓变背景估计。
        background=cv2.morphologyEx(work,cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
        background=cv2.GaussianBlur(background,(0,0),max(1,k/5))
        normalized=np.clip(work.astype(np.float32)*245/np.maximum(background.astype(np.float32),32),0,255).astype(np.uint8)
    enhanced=normalized.copy()
    if cfg.clahe_clip>0:
        enhanced=cv2.createCLAHE(clipLimit=cfg.clahe_clip,tileGridSize=(8,8)).apply(enhanced)
    if cfg.mode=='otsu':
        _,binary=cv2.threshold(enhanced,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        window=_odd_window(cfg.adaptive_block,min(enhanced.shape))
        binary=cv2.adaptiveThreshold(enhanced,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,window,cfg.adaptive_c)
    return {'gray':gray,'background':background,'normalized':normalized,
            'enhanced':enhanced,'binary':binary}


def describe_enhancement(config=None):
    cfg=config or EnhancementConfig()
    return {'config':asdict(cfg),'method':{
        'otsu':'共享去噪 + 全局Otsu阈值',
        'adaptive':'共享去噪 + 高斯自适应阈值',
        'illumination':'共享去噪 + 闭运算/高斯背景估计 + 除法补偿 + 高斯自适应阈值'}[cfg.mode],
        'notes':'CLAHE默认关闭；未默认对文字做腐蚀/膨胀。大面积图形、强遮挡、折痕不属于光照模型可恢复范围。'}
