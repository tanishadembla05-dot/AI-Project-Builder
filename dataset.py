import json
from pathlib import Path
from typing import Any, Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _load_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_project_templates() -> List[Dict[str, Any]]:
    return _load_json(DATA_DIR / "project_templates.json", [])


def load_tech_stacks() -> Dict[str, Dict[str, List[str]]]:
    return _load_json(DATA_DIR / "tech_stacks.json", {})


def load_risk_database() -> Dict[str, List[Dict[str, str]]]:
    return _load_json(DATA_DIR / "risk_database.json", {})


def get_template_by_id(template_id: int) -> Optional[Dict[str, Any]]:
    templates = load_project_templates()
    return next((tpl for tpl in templates if tpl.get("id") == template_id), None)
