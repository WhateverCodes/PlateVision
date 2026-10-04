"""Run independent pipelines sequentially on identical input pixels."""
from dataclasses import replace
import time
from .pipeline import Pipeline
from .pretrained import PretrainedDetector
from .whole_line_reader import WholeLineReader
from .classical_detector import PlateCropDetector


def build_pipelines(settings, cropped=False, learned_own=False):
    own=replace(settings,character_path=settings.adapted_character_path,reader_mode='characters',
                localization_mode='crop' if cropped else ('learned' if learned_own else 'pretrained'),
                rectify=False if cropped else settings.rectify)
    external=replace(settings,reader_mode='whole_line',rectify=False,localization_mode='pretrained',character_threshold=settings.ocr_threshold)
    detector=PlateCropDetector() if cropped else PretrainedDetector(settings.pretrained_folder,settings.plate_threshold,settings.cpu_threads)
    return {
        'My model':Pipeline(own,detector=None if learned_own and not cropped else detector),
        'Pretrained model':Pipeline(external,detector=detector,classifier=WholeLineReader(settings.whole_line_folder)),
    }


def compare(image,pipelines):
    results={}
    for name,pipe in pipelines.items():
        start=time.perf_counter()
        detections=sorted(pipe(image.copy()),key=lambda d:d.get('confidence',0.),reverse=True)
        results[name]={'detections':detections,'seconds':time.perf_counter()-start,
                       'detector':pipe.detector.provenance,'reader':pipe.classifier.provenance}
    return results


def reference_counts(detections,expected):
    """Distinct registrations only. Incorrect/extra accepted strings remain counted."""
    observed={d['raw_text'] for d in detections if d['recognized']}
    return {'correct':len(observed & expected),'missed':len(expected-observed),
            'wrong':len(observed-expected)}
