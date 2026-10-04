"""Session-only unique registry. One event per frame, never one per vehicle."""
from dataclasses import dataclass, field
from .validator import normalize, same_plate

def timestamp(seconds: float) -> str:
    return f"{int(seconds)//60:02d}:{int(seconds)%60:02d}"

@dataclass
class Entry:
    serial: str
    plate: str
    first_seconds: float
    confidence: float
    vehicle_type: str

@dataclass
class Event:
    id: int
    seconds: float
    frame_number: int
    fps: float
    serials: list[str]
    new_serials: list[str]
    image: bytes = b""

@dataclass
class Registry:
    fuzzy_threshold: float = 1.0
    entries: dict[str, Entry] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)

    def reset(self) -> None:
        self.entries.clear()
        self.events.clear()

    def observe(self, detections: list[dict], seconds: float = 0,
                frame_number: int = 0, fps: float = 1) -> Event | None:
        visible, new = [], []
        for d in detections:
            if not d.get("recognized", False):
                continue
            plate = normalize(d["text"])
            matches = [e for e in self.entries.values() if same_plate(e.plate, plate, self.fuzzy_threshold)]
            exact = self.entries.get(plate)
            entry = exact or (matches[0] if len(matches) == 1 else None)
            if entry is None:
                entry = Entry(f"{len(self.entries)+1:03d}", plate, seconds,
                              float(d["confidence"]), d.get("vehicle_type", "Unknown"))
                self.entries[plate] = entry
                new.append(entry.serial)
            d["serial"] = entry.serial
            if entry.serial not in visible:
                visible.append(entry.serial)
        if not new:
            return None
        event = Event(len(self.events), seconds, frame_number, fps, visible, new)
        self.events.append(event)
        return event

    def rows(self, video: bool = False, event_id: int | None = None) -> list[dict]:
        def row(e: Entry, seconds: float | None = None) -> dict:
            out = {"Sr No.": e.serial, "Vehicle Type": e.vehicle_type, "License Plate": e.plate}
            if seconds is not None:
                out["Timestamp"] = timestamp(seconds)
            return out
        if not video:
            return [row(e) for e in self.entries.values()]
        by_serial = {e.serial:e for e in self.entries.values()}
        return [row(by_serial[s], ev.seconds) for ev in self.events
                if event_id is None or ev.id == event_id for s in ev.serials]
