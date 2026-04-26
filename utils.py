import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def clean_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return [token for token in cleaned.split() if len(token) > 2]


def keyword_similarity(a: str, b: str) -> float:
    a_tokens = Counter(tokenize(a))
    b_tokens = Counter(tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = sum((a_tokens & b_tokens).values())
    union = sum((a_tokens | b_tokens).values())
    return intersection / union if union else 0.0


def unique_preserve(items: Iterable[Any]) -> List[Any]:
    seen = set()
    output = []
    for item in items:
        marker = item
        if isinstance(item, (dict, list)):
            marker = json.dumps(item, sort_keys=True)
        if marker not in seen:
            output.append(item)
            seen.add(marker)
    return output


def ensure_daywise_non_repetitive(day_plan: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    seen_tasks = set()
    for day_key, payload in day_plan.items():
        tasks = payload.get("tasks", [])
        updated = []
        for task in tasks:
            if isinstance(task, dict):
                base_summary = str(task.get("summary", "Task")).strip() or "Task"
                summary = base_summary
                suffix = 1
                while summary.lower() in seen_tasks:
                    suffix += 1
                    summary = f"{base_summary} (variation {suffix})"
                seen_tasks.add(summary.lower())
                task["summary"] = summary
                updated.append(task)
            else:
                base = str(task).strip() or "Task"
                candidate = base
                suffix = 1
                while candidate.lower() in seen_tasks:
                    suffix += 1
                    candidate = f"{base} (variation {suffix})"
                seen_tasks.add(candidate.lower())
                updated.append(candidate)
        payload["tasks"] = updated
        day_plan[day_key] = payload
    return day_plan


def make_report_summary(project_name: str, category: str, difficulty: int, confidence: int) -> str:
    return (
        f"This roadmap outlines a {category} project plan for '{project_name}' with an estimated "
        f"difficulty of {difficulty}/10 and completion confidence of {confidence}%. "
        "The plan is structured for steady delivery, incremental validation, and final demo readiness."
    )


def safe_json_loads(raw_text: str, default: Any) -> Any:
    try:
        return json.loads(raw_text)
    except Exception:
        return default


def save_plan_locally(plan: Dict[str, Any], base_dir: Path) -> Path:
    save_dir = base_dir / "saved_plans"
    save_dir.mkdir(exist_ok=True, parents=True)
    project_name = re.sub(r"[^a-zA-Z0-9_-]", "_", plan.get("project_idea", "project"))[:40]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = save_dir / f"{project_name}_{stamp}.json"
    out_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return out_file


def list_saved_plans(base_dir: Path) -> List[Path]:
    save_dir = base_dir / "saved_plans"
    if not save_dir.exists():
        return []
    return sorted(save_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def timeline_phase(total_days: int, day_num: int) -> str:
    if total_days <= 3:
        return "Execution"
    if total_days <= 14:
        boundaries = [max(1, int(total_days * ratio)) for ratio in (0.2, 0.65, 0.85, 1.0)]
        if day_num <= boundaries[0]:
            return "Planning"
        if day_num <= boundaries[1]:
            return "Development"
        if day_num <= boundaries[2]:
            return "Testing"
        return "Polish & Demo"
    boundaries = [max(1, int(total_days * ratio)) for ratio in (0.15, 0.6, 0.8, 0.92, 1.0)]
    if day_num <= boundaries[0]:
        return "Planning"
    if day_num <= boundaries[1]:
        return "Development"
    if day_num <= boundaries[2]:
        return "Testing"
    if day_num <= boundaries[3]:
        return "Polish"
    return "Demo"
