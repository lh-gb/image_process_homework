"""几何正确性、空白失败路径和有已知真值的合成拍摄回归测试。"""
import unittest
import cv2
import numpy as np
from docscan.config import DetectionConfig, EnhancementConfig
from docscan.detection import detect_document
from docscan.geometry import polygon_iou, corner_error, rectify_document
from docscan.pipeline import process_document, ProcessingError
from docscan.enhancement import enhance_document

class DocumentTests(unittest.TestCase):
    def test_known_projective_document(self):
        page=np.full((560,400,3),242,np.uint8)
        for y in range(45,520,22): cv2.line(page,(25,y),(365,y),(35,35,35),2)
        page[:15]=242; page[-15:]=242
        gt=np.float32([[180,70],[610,100],[570,650],[130,610]])
        transform=cv2.getPerspectiveTransform(np.float32([[0,0],[399,0],[399,559],[0,559]]),gt)
        canvas=np.full((720,820,3),(95,110,125),np.uint8)
        warped=cv2.warpPerspective(page,transform,(820,720))
        mask=cv2.warpPerspective(np.full(page.shape[:2],255,np.uint8),transform,(820,720))
        canvas[mask>0]=warped[mask>0]
        pred=detect_document(canvas,DetectionConfig())
        self.assertIsNotNone(pred.corners)
        self.assertGreater(polygon_iou(pred.corners,gt),.94)
        self.assertLess(corner_error(pred.corners,gt,canvas.shape),.015)
        restored=rectify_document(canvas,gt,output_size=(400,560))
        self.assertEqual(restored.shape,(560,400,3))
        self.assertLess(np.abs(restored.astype(float)-page).mean(),15)

    def test_blank_fails_explicitly_and_manual_recovers(self):
        image=np.full((300,400,3),180,np.uint8)
        with self.assertRaises(ProcessingError): process_document(image)
        r=process_document(image,corners=[[20,20],[380,20],[380,280],[20,280]])
        self.assertEqual(r.metadata['corner_source'],'manual')
        self.assertEqual(r.binary.dtype,np.uint8)

    def test_shadow_correction_on_known_blank_background(self):
        x=np.linspace(.48,1,400)[None,:]
        gray=np.repeat((245*x).astype(np.uint8),300,axis=0)
        cv2.putText(gray,'Document',(30,160),cv2.FONT_HERSHEY_SIMPLEX,1.2,20,2)
        image=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        out=enhance_document(image,EnhancementConfig(mode='illumination',denoise=False))
        # 无文字的顶端区域：光照梯度应减小，文字区域仍存在黑色像素。
        self.assertLess(float(out['enhanced'][:50].std()),float(gray[:50].std())*.65)
        self.assertGreater(np.count_nonzero(out['binary'][120:175]==0),100)

    def test_corner_metric_is_invariant_to_cyclic_start(self):
        p=np.float32([[10,10],[100,20],[90,160],[0,140]])
        self.assertAlmostEqual(polygon_iou(p,np.roll(p,2,axis=0)),1,places=5)
        self.assertAlmostEqual(corner_error(p,p[::-1],(200,200)),0,places=6)
        with self.assertRaises(ValueError): rectify_document(np.zeros((20,20,3),np.uint8),[[0,0]]*4)

    def test_small_constant_image_enhancement(self):
        for size in (8,16,32):
            for mode in ('otsu','adaptive','illumination'):
                out=enhance_document(np.full((size,size,3),210,np.uint8),EnhancementConfig(mode=mode))
                self.assertEqual(out['binary'].shape,(size,size))
                self.assertTrue(set(np.unique(out['binary'])).issubset({0,255}))

if __name__=='__main__': unittest.main()
