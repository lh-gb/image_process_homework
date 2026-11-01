from dataclasses import dataclass

@dataclass(frozen=True)
class DetectionConfig:
    mode: str = 'robust'
    max_dimension: int = 1280
    min_area_ratio: float = 0.04
    max_area_ratio: float = 0.94
    use_normalized_edges: bool = True
    use_threshold_proposals: bool = True
    scoring: str = 'combined'

    def __post_init__(self):
        if self.mode not in ('baseline', 'robust'):
            raise ValueError('mode 必须为 baseline 或 robust')
        if self.scoring not in ('combined', 'area'):
            raise ValueError('scoring 必须为 combined 或 area')
        if self.max_dimension < 128:
            raise ValueError('检测分辨率至少为 128')
        if not 0 < self.min_area_ratio < self.max_area_ratio < 1:
            raise ValueError('面积比例必须满足 0 < min < max < 1')

@dataclass(frozen=True)
class EnhancementConfig:
    mode: str = 'illumination'
    denoise: bool = True
    background_kernel: int = 0
    adaptive_block: int = 41
    adaptive_c: float = 11.0
    clahe_clip: float = 0.0

    def __post_init__(self):
        if self.mode not in ('otsu', 'adaptive', 'illumination'):
            raise ValueError('未知增强模式')
        if self.background_kernel < 0 or self.adaptive_block < 3:
            raise ValueError('背景核须非负，自适应窗口至少为 3')
        if self.clahe_clip < 0:
            raise ValueError('CLAHE clip 必须非负')
