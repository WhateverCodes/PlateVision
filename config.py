"""Conservative defaults for an Intel Ultra 7 laptop; no automatic training."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

@dataclass(frozen=True)
class Settings:
    image_size: int = 320  # 640 is an optional quality profile; divisible by 32
    cpu_threads: int = 2
    max_video_seconds: float = 30.0
    max_upload_mb: int = 100
    max_image_pixels: int = 24_000_000
    max_video_pixels: int = 3840 * 2160
    max_decode_frames: int = 9000
    max_events: int = 100
    target_fps: float = 2.0
    frame_skip: int = 1
    plate_threshold: float = 0.5
    character_threshold: float = 0.70
    ocr_threshold: float = 0.91
    vehicle_threshold: float = 0.5
    fuzzy_threshold: float = 1.0  # exact by default; optional approximate matching
    nms_iou: float = 0.4
    clahe: bool = False
    bilateral: bool = False
    sharpen: bool = False
    rectify: bool = True
    reader_mode: str = "characters"
    pretrained_folder: Path = ROOT / "models/pretrained"
    whole_line_folder: Path = next((ROOT / p for p in ("models/paddle_combined_v2", "models/paddle_plate_adapted_v1", "models/paddle_candidate") if (ROOT / p / "english.onnx").exists()), ROOT / "models/paddle_candidate")
    localization_mode: str = "learned"
    reader_path: Path = ROOT / "models/plate_reader_pilot/best.pt"
    detector_path: Path = ROOT / ("models/plate_detector_real_v3/best.pt" if (ROOT / "models/plate_detector_real_v3/best.pt").exists() else "models/plate_detector/best.pt")
    adapted_character_path: Path = next((ROOT / p for p in ("models/character_conservative_v1/last.pt", "models/character_verified_100/last.pt", "models/character_adaptation_v1/last.pt") if (ROOT / p).exists()), ROOT / "models/character_adaptation_v1/last.pt")
    character_path: Path = ROOT / "models/character_classifier/best.pt"
    vehicle_path: Path = ROOT / "models/vehicle_detector/ssdlite320.pth"

DEFAULTS = Settings()
