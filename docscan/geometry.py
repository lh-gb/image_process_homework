"""四边形几何；内部统一顺时针绕序，起点选择几何左上附近。"""
import cv2
import numpy as np


def order_points(points):
    p = np.asarray(points, dtype=np.float32)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        raise ValueError('角点必须为有限的 4×2 坐标')
    hull = cv2.convexHull(p).reshape(-1, 2)
    if len(hull) != 4 or abs(cv2.contourArea(hull)) < 4:
        raise ValueError('四角必须构成非退化凸四边形')
    c = hull.mean(axis=0)
    p = hull[np.argsort(np.arctan2(hull[:, 1]-c[1], hull[:, 0]-c[0]))]
    p = np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)
    return np.ascontiguousarray(p, dtype=np.float32)


def rectify_document(image, corners, output_size=None):
    p = order_points(corners)
    if output_size is None:
        w = round(max(np.linalg.norm(p[1]-p[0]), np.linalg.norm(p[2]-p[3])))
        h = round(max(np.linalg.norm(p[3]-p[0]), np.linalg.norm(p[2]-p[1])))
    else:
        w, h = map(int, output_size)
    if min(w, h) < 8 or max(w, h) > 16000 or w*h > 50_000_000:
        raise ValueError('展正图像尺寸无效或过大')
    dst = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    matrix = cv2.getPerspectiveTransform(p, dst)
    return cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def polygon_iou(a, b):
    try:
        a, b = order_points(a), order_points(b)
        inter, _ = cv2.intersectConvexConvex(a, b)
        union = abs(cv2.contourArea(a)) + abs(cv2.contourArea(b)) - inter
        return float(np.clip(inter / union, 0, 1)) if union > 0 else 0.0
    except (ValueError, cv2.error, TypeError):
        return 0.0


def corner_error(pred, gt, image_shape):
    """循环配对，平均角点欧氏距离 / 原图对角线；无效预测由调用方单独记失败。"""
    a, b = order_points(pred), order_points(gt)
    error = min(np.linalg.norm(a-np.roll(b, i, axis=0), axis=1).mean() for i in range(4))
    return float(error / np.hypot(image_shape[0], image_shape[1]))
