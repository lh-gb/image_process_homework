"""多路候选与可解释评分。检测器不接收任何标注或文档身份。"""
from dataclasses import dataclass, field
import cv2
import numpy as np
from .config import DetectionConfig
from .geometry import order_points

@dataclass
class DetectionResult:
    corners: np.ndarray | None
    score: float = 0.0
    method: str = ''
    candidate_count: int = 0
    diagnostics: dict = field(default_factory=dict)
    edges: np.ndarray | None = None


def _valid(q, shape, cfg):
    h, w = shape[:2]
    area = abs(cv2.contourArea(q)) / (w*h)
    if not cfg.min_area_ratio <= area <= cfg.max_area_ratio:
        return False
    edges = np.roll(q, -1, axis=0)-q
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.min() < 16 or lengths.max()/lengths.min() > 12:
        return False
    # 排除近共线退化角；不强制直角，允许正常透视变形。
    for i in range(4):
        u, v = q[(i-1)%4]-q[i], q[(i+1)%4]-q[i]
        cosine = np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)+1e-6)
        if abs(cosine) > 0.97:
            return False
    # 大面积碰到图像边框的候选通常是整幅画面/桌面边界。
    if sum(x < 3 or y < 3 or x > w-4 or y > h-4 for x,y in q) >= 2:
        return False
    return True


def _proposals(binary, cfg):
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:100]:
        if abs(cv2.contourArea(cnt)) < binary.size*cfg.min_area_ratio*0.6:
            continue
        for contour in (cnt, cv2.convexHull(cnt)):
            perimeter = cv2.arcLength(contour, True)
            for eps in (0.012, 0.02, 0.035, 0.055):
                p = cv2.approxPolyDP(contour, eps*perimeter, True)
                if len(p) != 4 or not cv2.isContourConvex(p):
                    continue
                try:
                    q = order_points(p.reshape(4,2))
                    if _valid(q, binary.shape, cfg):
                        result.append(q)
                except ValueError:
                    continue
    return result


def _features(q, lab, edge_distance):
    h,w = lab.shape[:2]
    mask = np.zeros((h,w), np.uint8)
    cv2.fillConvexPoly(mask, q.astype(np.int32), 255)
    inner = cv2.erode(mask, np.ones((9,9),np.uint8)) > 0
    outer = (cv2.dilate(mask,np.ones((13,13),np.uint8)) > 0) & (mask == 0)
    inside = lab[inner]
    outside = lab[outer]
    if not len(inside) or not len(outside):
        return None
    # 内外色差不假定白纸一定比桌面亮；亮度先验仅作弱项。
    color_gap = float(np.linalg.norm(np.median(inside,axis=0)-np.median(outside,axis=0)))
    supports = []
    for i in range(4):
        t = np.linspace(.06,.94,48)[:,None]
        p = q[i]*(1-t) + q[(i+1)%4]*t
        x = np.clip(np.rint(p[:,0]).astype(int),0,w-1)
        y = np.clip(np.rint(p[:,1]).astype(int),0,h-1)
        supports.append(float(np.exp(-edge_distance[y,x]/2.5).mean()))
    support = .65*np.mean(supports)+.35*min(supports)
    area = abs(cv2.contourArea(q))/(w*h)
    brightness = float(np.median(inside[:,0])/255)
    center = 1 - min(float(np.linalg.norm(q.mean(0)-[w/2,h/2]) / np.hypot(w/2,h/2)),1)
    # 面积加分饱和，避免最大桌面轮廓压过文档；权重固定、仅作排序分数。
    score = (.42*support + .30*min(color_gap/45,1) + .12*brightness
             + .10*min(area/.18,1) + .06*center)
    return {'score':float(score),'boundary_support':float(support),
            'color_gap':color_gap,'area_ratio':float(area),'brightness':brightness}


def detect_document(image, config=None):
    cfg = config or DetectionConfig()
    if image is None or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 16:
        raise ValueError('检测输入应为至少16×16的BGR图像')
    h,w = image.shape[:2]
    scale = min(cfg.max_dimension/max(h,w),1.0)
    small = cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    basic_edges = cv2.Canny(blur,75,200)
    if cfg.mode == 'baseline':
        contours,_ = cv2.findContours(basic_edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        choices=[]
        for cnt in contours:
            p=cv2.approxPolyDP(cnt,.02*cv2.arcLength(cnt,True),True)
            if len(p)==4 and cv2.isContourConvex(p):
                try:
                    q=order_points(p.reshape(4,2))
                    if _valid(q,gray.shape,cfg): choices.append(q)
                except ValueError: pass
        if not choices:
            return DetectionResult(None,method='baseline',edges=basic_edges,
                                   diagnostics={'reason':'no_valid_quadrilateral'})
        q=max(choices,key=lambda p:abs(cv2.contourArea(p)))
        return DetectionResult(q/scale,abs(cv2.contourArea(q))/gray.size,'baseline',len(choices),
                               {'score_kind':'area_ratio','scale':scale},basic_edges)
    maps=[]
    for low,high in ((35,110),(60,180),(90,230)):
        edge=cv2.Canny(blur,low,high)
        maps.append((f'gray_canny_{low}',cv2.morphologyEx(edge,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))))
    if cfg.use_normalized_edges:
        # 大尺度背景归一化可减弱缓变照明对边缘的影响。
        bg=cv2.GaussianBlur(blur,(0,0),max(12,min(gray.shape)/18))
        norm=cv2.divide(blur,np.maximum(bg,1),scale=180)
        maps.append(('normalized_canny',cv2.morphologyEx(cv2.Canny(norm,40,120),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))))
    if cfg.use_threshold_proposals:
        lab=cv2.cvtColor(small,cv2.COLOR_BGR2LAB)
        for channel in (lab[:,:,0],gray,lab[:,:,1],lab[:,:,2]):
            _,mask=cv2.threshold(channel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
            maps.append(('otsu_region',mask))
            maps.append(('otsu_inverse_region',cv2.morphologyEx(255-mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))))
    candidates=[]
    seen=set()
    for name,binary in maps:
        for q in _proposals(binary,cfg):
            key=tuple(np.rint(q.reshape(-1)/4).astype(int))
            if key not in seen:
                seen.add(key); candidates.append((q,name))
    combined=basic_edges.copy()
    for _,m in maps:
        # 区域掩膜仅参与候选生成，不将其内部视为边缘。
        combined=cv2.bitwise_or(combined,cv2.Canny(m,50,150))
    distance=cv2.distanceTransform(255-combined,cv2.DIST_L2,3)
    lab=cv2.cvtColor(small,cv2.COLOR_BGR2LAB)
    scored=[]
    for q,name in candidates:
        f=_features(q,lab,distance)
        if f is not None:
            score=f['area_ratio'] if cfg.scoring=='area' else f['score']
            scored.append((score,q,name,f))
    if not scored:
        return DetectionResult(None,method='robust',candidate_count=len(candidates),edges=combined,
                               diagnostics={'reason':'no_valid_quadrilateral','scale':scale})
    scored.sort(key=lambda item:item[0],reverse=True)
    score,q,name,features=scored[0]
    return DetectionResult(q/scale,float(score),'robust',len(candidates),
                           {'proposal':name,'features':features,'scale':scale,
                            'score_kind':'ranking_score_not_probability',
                            'top_candidates':[{'score':float(s),'proposal':n,'corners':(p/scale).tolist()}
                                              for s,p,n,_ in scored[:5]]},combined)
