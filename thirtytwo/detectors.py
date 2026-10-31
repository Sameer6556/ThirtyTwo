"""The YOLO models, each run the way its reference in assets/ was made.

* Tooth detector: teeth_clean.pkl came from model.predict on chunks of 16 mixed-size X-rays (cell 18, yolo_boxes), so
  ultralytics padded nearly every X-ray to a full 1024 square. predict(rect=False) does the same for one X-ray.
* Disease detectors: the disbox_*_validator.pkl files came from model.val (cell 18, validator_boxes), which pads to
  rectangular batch shapes with half a stride extra and uses multi-label NMS. DiseaseDetector copies the validator for
  a batch of one: BaseDataset.load_image, set_rectangle, LetterBox, Format, DetectionValidator.preprocess and
  postprocess, scale_preds and pred_to_json (its rounding), then the notebook's division by the export scale.
"""
import math

import numpy as np
import torch

from . import config as C


def _device_arg(device):
    return 0 if device.type == 'cuda' else 'cpu'


class ToothDetector:
    """YOLO11s, 32 classes (one per FDI tooth number), at 1024 px; the most confident box per tooth number"""

    def __init__(self, weights, device):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.device = device

    def __call__(self, bgr, scale):
        import dentex_utils as U
        t = C.TOOTH
        r = self.model.predict(source=bgr, imgsz=t['imgsz'], conf=t['conf'], iou=t['iou'],
                               agnostic_nms=t['agnostic_nms'], max_det=t['max_det'], rect=False,
                               device=_device_arg(self.device), verbose=False)[0]
        b = r.boxes
        return U.teeth_from_yolo(b.xyxy.cpu().numpy() / scale, b.cls.cpu().numpy().astype(int),
                                 b.conf.cpu().numpy())


class DiseaseDetector:
    """a 4-class YOLO, boxes as ultralytics' validator makes them (conf 0.001, NMS IoU 0.7, up to 300, multi-label)"""

    def __init__(self, weights, imgsz, device):
        from ultralytics import YOLO
        from ultralytics.nn.autobackend import AutoBackend
        from ultralytics.utils.checks import check_imgsz
        model = YOLO(str(weights)).model
        # as BaseValidator.__call__ builds it: nms=None -> end2end=False, quantize=None -> fp32, channels_last auto
        # (verbose only silences the layer summary it prints while fusing)
        self.backend = AutoBackend(model=model, device=device, dnn=False, data=None, fp16=False, verbose=False,
                                   channels_last=None, end2end=False)
        self.backend.eval()
        self.device = self.backend.device
        self.stride = self.backend.stride
        self.imgsz = check_imgsz(imgsz, stride=self.stride)
        self.end2end = getattr(self.backend, 'end2end', False)   # DetectionValidator.init_metrics

    def batch_shape(self, h0, w0):
        """BaseDataset.set_rectangle for a batch of one X-ray (pad 0.5 in val mode)"""
        ar = np.array([h0 / w0])
        mini, maxi = ar.min(), ar.max()
        shape = [1, 1]
        if maxi < 1:
            shape = [maxi, 1]
        elif mini > 1:
            shape = [1, 1 / mini]
        return np.ceil(np.array([shape]) * self.imgsz / self.stride + 0.5).astype(int)[0] * self.stride

    def letterbox(self, bgr):
        """BaseDataset.get_image_and_label + load_image (rect mode) + LetterBox + Format, for one image"""
        import cv2
        from ultralytics.data.augment import LetterBox
        h0, w0 = bgr.shape[:2]
        im = bgr
        r = self.imgsz / max(h0, w0)
        if r != 1:
            w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        label = dict(img=im, ori_shape=(h0, w0), resized_shape=im.shape[:2],
                     ratio_pad=(im.shape[0] / h0, im.shape[1] / w0), rect_shape=self.batch_shape(h0, w0))
        label = LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)(label)
        img = np.ascontiguousarray(label['img'].transpose(2, 0, 1)[::-1])      # HWC BGR -> CHW RGB
        return torch.from_numpy(img), (h0, w0), label['ratio_pad']

    @torch.inference_mode()
    def __call__(self, bgr, scale):
        """boxes in ORIGINAL pixels: [dict(box=[x1, y1, x2, y2], cls, conf)], as validator_boxes returns them"""
        from ultralytics.utils import nms, ops
        img, ori_shape, ratio_pad = self.letterbox(bgr)
        x = img[None].to(self.device).float() / 255
        preds = self.backend(x)
        p = nms.non_max_suppression(preds, C.DET_EVAL['conf'], C.DET_EVAL['iou'], nc=0, multi_label=True,
                                    agnostic=False, max_det=C.DET_EVAL['max_det'], end2end=self.end2end,
                                    rotated=False)[0]
        boxes = ops.scale_boxes(x.shape[2:], p[:, :4].clone(), ori_shape, ratio_pad=ratio_pad)
        xywh = ops.xyxy2xywh(boxes)
        xywh[:, :2] -= xywh[:, 2:] / 2                                          # pred_to_json
        out = []
        for b, s, c in zip(xywh.tolist(), p[:, 4].tolist(), p[:, 5].tolist()):
            bx, by, bw, bh = [round(v, 3) for v in b]
            out.append(dict(box=[bx / scale, by / scale, (bx + bw) / scale, (by + bh) / scale], cls=int(c),
                            conf=float(round(s, 5))))
        return out
