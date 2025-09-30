"""
ROI Configuration Loader

Loads Region-Of-Interest rectangles from JSON files in /home/hale/hale/exports
and provides a tracker-friendly mapping.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Optional


EXPORTS_DIR = Path("/home/hale/hale/exports")


def _load_roi_from_json_file(file_path: Path) -> Optional[Tuple[int, int, int, int]]:
    """
    Load an ROI rectangle from a JSON file of the form:
    { "roi": { "type": "rectangle", "x1": int, "y1": int, "x2": int, "y2": int } }
    """
    try:
        with file_path.open("r") as f:
            data = json.load(f)
        roi = data.get("roi", {})
        if not roi:
            return None
        return (
            int(roi.get("x1", 0)),
            int(roi.get("y1", 0)),
            int(roi.get("x2", 0)),
            int(roi.get("y2", 0)),
        )
    except Exception:
        return None


def load_all_export_rois() -> Dict[str, Tuple[int, int, int, int]]:
    """Load all known ROI files from /exports and return a name->coords map."""
    files = {
        "break_line_ROI_current_phase": EXPORTS_DIR / "break_line_ROI_current_phase.json",
        "break_line_ROI_return_phase": EXPORTS_DIR / "break_line_ROI_return_phase.json",
        "break_line_ROI_dish_count": EXPORTS_DIR / "break_line_ROI_dish_count.json",
        "kitchen_dish_ROI_dish_count": EXPORTS_DIR / "kitchen_dish_ROI_dish_count.json",
    }

    rois: Dict[str, Tuple[int, int, int, int]] = {}
    for name, path in files.items():
        coords = _load_roi_from_json_file(path)
        if coords is not None:
            rois[name] = coords
    return rois


def get_tracker_roi_config() -> Dict[str, Tuple[int, int, int, int]]:
    """
    Return a tracker-friendly ROI mapping using the exports:
    - dish_detection: ROI where break-line dish crossings are counted
    - incoming_phase: ROI to visualize/detect current phase (break-line)
    - return_phase: ROI to visualize/detect return phase (break-line)
    - kitchen_counter: ROI where kitchen dish crossings are counted
    """
    exported = load_all_export_rois()

    # Fallbacks in case files are missing (kept empty so drawing simply skips)
    dish_detection = exported.get("break_line_ROI_dish_count", (0, 0, 0, 0))
    incoming_phase = exported.get("break_line_ROI_current_phase", (0, 0, 0, 0))
    return_phase = exported.get("break_line_ROI_return_phase", (0, 0, 0, 0))
    kitchen_counter = exported.get("kitchen_dish_ROI_dish_count", (0, 0, 0, 0))

    return {
        "dish_detection": dish_detection,
        "incoming_phase": incoming_phase,
        "return_phase": return_phase,
        "kitchen_counter": kitchen_counter,
    }


