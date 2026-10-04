from pathlib import Path
import hashlib,json
p=Path('outputs/PlateVision1');keys=['models/character_conservative_v1/last.pt','models/paddle_plate_adapted_v1/english.onnx','models/plate_detector/best.pt','models/pretrained/yolo-v9-t-384-license-plates-end2end.onnx'];value={k:hashlib.file_digest((p/k).open('rb'),'sha256').hexdigest() for k in keys};(p/'outputs/evaluation/combined_baseline_weights.json').write_text(json.dumps(value,indent=2));print('Retained baseline fingerprints recorded.')
