"""Complete image -> plate -> segmented characters -> text pipeline."""
import numpy as np
from .preprocessing import grayscale,enhance,rectify_plate
from .character_segmenter import segment
from .validator import normalize,plausible,plausible_ocr
from .association import associate

class Pipeline:
    def __init__(self,settings,detector=None,classifier=None,vehicle_detector=None):
        self.settings = settings
        if detector is None:
            if settings.localization_mode=="pretrained":
                from .pretrained import PretrainedDetector
                detector=PretrainedDetector(settings.pretrained_folder,settings.plate_threshold,settings.cpu_threads)
            elif settings.localization_mode=="classical":
                from .classical_detector import ClassicalDetector
                detector=ClassicalDetector()
            elif settings.localization_mode=="crop":
                from .classical_detector import PlateCropDetector
                detector=PlateCropDetector()
            else:
                from .plate_detector import Detector
                detector = Detector(settings.detector_path,settings.plate_threshold,settings.nms_iou)
        if classifier is None:
            if settings.reader_mode=="whole_line":
                from .whole_line_reader import WholeLineReader
                classifier=WholeLineReader(settings.whole_line_folder)
            elif settings.reader_mode=="whole_plate":
                from .plate_reader import Reader
                classifier=Reader(settings.reader_path)
            else:
                from .character_classifier import Classifier
                classifier = Classifier(settings.character_path)
        self.detector,self.classifier,self.vehicles = detector,classifier,vehicle_detector

    def __call__(self,image):
        cfg = self.settings
        gray = enhance(grayscale(image),cfg.clahe,cfg.bilateral,cfg.sharpen)
        vehicles = self.vehicles(image) if self.vehicles else []
        results = []
        for detection in self.detector(image if cfg.localization_mode=="pretrained" else gray):
            x1,y1,x2,y2 = detection["box"]
            crop = image[y1:y2,x1:x2]
            if crop.size == 0:
                continue
            plate = rectify_plate(crop) if cfg.rectify else crop
            if cfg.reader_mode in {"whole_plate","pretrained","whole_line"}:
                import cv2
                chars=[]
                binary=cv2.threshold(grayscale(plate),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            else:
                chars,binary = segment(plate)
            if cfg.reader_mode in {"whole_plate","pretrained","whole_line"}:
                text,confidences=self.classifier.read(plate)
            elif hasattr(self.classifier,"read_segmented"):
                chars,binary,text,confidences=self.classifier.read_segmented(plate,cfg.character_threshold)
            else:
                text,confidences=self.classifier(chars)
            text = normalize(text)
            # PaddleOCR uses mean emitted-token confidence; our independent
            # character classifier retains its existing minimum-glyph gate.
            confidence = float(np.mean(confidences) if cfg.reader_mode=="whole_line" else min(confidences)) if confidences else 0.0
            accepted = confidence >= cfg.character_threshold and (plausible_ocr(text) if cfg.reader_mode=='whole_line' else plausible(text))
            vehicle = associate(detection["box"],vehicles)
            results.append({**detection,"reader_mode":cfg.reader_mode,"text":text if accepted else "Not recognized",
                "raw_text":text,"confidence":confidence,"recognized":accepted,
                "confidence_method":"mean OCR token score" if cfg.reader_mode=="whole_line" else "minimum character score",
                "vehicle_type":vehicle["vehicle_type"] if vehicle else "Unknown",
                "vehicle_box":vehicle["box"] if vehicle else None,
                "crop":plate,"grayscale":grayscale(plate),"binary":binary,"characters":chars,
                "character_confidences":confidences})
        return results
