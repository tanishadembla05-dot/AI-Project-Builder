from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import streamlit as st
import streamlit.components.v1 as components

from agent import run_agent
from llm_client import LLMClient
from pdf_exporter import PDFExporter
from utils import list_saved_plans, safe_json_loads, save_plan_locally


BASE_DIR = Path(__file__).resolve().parent
TECH_ICONS = {
    "python": "🐍",
    "react": "⚛️",
    "flutter": "🎯",
    "firebase": "🔥",
    "docker": "🐳",
    "streamlit": "📊",
    "langgraph": "🧠",
    "fastapi": "⚡",
    "mongodb": "🍃",
    "postgresql": "🐘",
    "redis": "🧰",
    "node.js": "🟢",
    "typescript": "🟦",
}


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp { background-color: #0d0d0d; color: #f5f5f5; font-family: "Inter", "Segoe UI", sans-serif; }
        .card { background: #1a1a2e; border: 1px solid #2a2a44; border-radius: 14px; padding: 14px; margin: 8px 0; transition: 0.25s; }
        .card:hover { box-shadow: 0 0 12px rgba(0,212,255,0.28); border-color: #00d4ff; }
        .badge { display:inline-block; padding:4px 10px; border-radius: 999px; background:#132838; color:#00d4ff; border:1px solid #00d4ff; font-size:12px; margin:4px 6px 4px 0; }
        .chip { display:inline-block; padding:5px 10px; border-radius: 999px; background:#102234; color:#d9f7ff; border:1px solid #1d4d69; margin:4px 6px 4px 0; }
        .success { color:#00ff88; }
        .warn { color:#ffaa00; }
        .danger { color:#ff4466; }
        .mono { font-family: "Consolas", "Courier New", monospace; }
        .tiny { font-size: 12px; opacity: 0.85; }
        .redo-card { background: #231b12; border: 1px solid #ffaa00; border-radius: 14px; padding: 14px; margin: 8px 0; }
        div[data-testid="stExpander"] { background: #141428; border: 1px solid #2a2a44; border-radius: 10px; margin-bottom: 8px; }
        div[data-testid="stExpander"] details[open] { background: #1b1b32; }
        div[data-testid="stProgressBar"] > div > div > div > div { background-color: #00d4ff; transition: all 0.4s ease; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def severity_class(severity: str) -> str:
    severity = (severity or "").lower()
    if severity == "high":
        return "danger"
    if severity == "low":
        return "success"
    return "warn"


def init_state():
    defaults = {
        "plan_result": None,
        "current_day": 1,
        "redo_tasks": {},
        "completed_tasks": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def normalize_task(task: Any) -> Dict[str, str]:
    if isinstance(task, dict):
        summary = str(task.get("summary", "")).strip() or "Task"
        return {
            "summary": summary,
            "what": str(task.get("what", summary)),
            "how": str(task.get("how", "")),
            "output": str(task.get("output", "")),
            "time": str(task.get("time", "45 mins")),
        }
    text = str(task).strip() or "Task"
    return {"summary": text, "what": text, "how": "", "output": "", "time": "45 mins"}


def day_completion_status(day_number: int, tasks: list[Any], redo_tasks: list[Dict[str, Any]]) -> tuple[int, int, bool]:
    day_key = f"day_{day_number}"
    total_count = len(tasks) + len(redo_tasks)
    completed_indices = set(st.session_state.get("completed_tasks", {}).get(day_key, []))
    done_count = len([idx for idx in range(len(tasks)) if idx in completed_indices])
    for idx in range(len(redo_tasks)):
        if st.session_state.get(f"redo_done_{day_number}_{idx}", False):
            done_count += 1
    is_day_done = total_count > 0 and done_count == total_count
    return done_count, total_count, is_day_done


def regenerate_missed_tasks(
    plan: Dict[str, Any], selected_day: int, total_days: int, missed_tasks: list[str]
) -> list[Dict[str, str]]:
    provider = str(plan.get("provider", "groq")).lower().strip()
    client = LLMClient(provider=provider)
    fallback = [
        {
            "task": f"Break down and complete: {task} -> Output: completed sub-task with proof in notes or terminal output",
            "duration": "1 hour",
            "tip": "Time-box this task to 45 minutes and keep a single clear acceptance check.",
        }
        for task in missed_tasks
    ]

    if not missed_tasks:
        return []
    if not client.is_configured():
        return fallback

    prompt = f"""
The user is working on: {plan.get("project_idea", "")}
Skill level: {plan.get("skill_level", "Beginner")}
They are on Day {selected_day} of {total_days}.

They did NOT complete these tasks today:
{missed_tasks}

Generate ONLY replacement tasks for these missed items.
Each replacement task must:
- Directly replace the missed task with a simpler or broken-down version
- Be completable in 1-2 hours max
- Be specific with clear output
- NOT repeat tasks from other days

Return as a JSON list:
[
  {{
    "task": "specific task description -> Output: concrete result",
    "duration": "1 hour",
    "tip": "one practical tip to complete this faster"
  }}
]
"""
    out = client.call_llm(prompt=prompt, system_prompt="You are a practical project mentor. Return JSON only.")
    parsed = safe_json_loads(out, fallback)
    if isinstance(parsed, list) and parsed:
        cleaned = []
        for item in parsed[: len(missed_tasks)]:
            if isinstance(item, dict):
                cleaned.append(
                    {
                        "task": str(item.get("task", "")),
                        "duration": str(item.get("duration", "1 hour")),
                        "tip": str(item.get("tip", "Break this into one small milestone first.")),
                    }
                )
        if cleaned:
            return cleaned
    return fallback


def load_selected_plan(file_path: Path):
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        st.session_state["plan_result"] = payload
        st.session_state["current_day"] = 1
        st.session_state["redo_tasks"] = {}
        st.session_state["completed_tasks"] = {}
        st.success(f"Loaded {file_path.name}")
    except Exception as exc:
        st.error(f"Could not load plan: {exc}")


def render_overview(plan: Dict[str, Any]):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### {plan.get('project_idea','Project')}")
    st.markdown(f"<span class='badge'>{plan.get('category','General')}</span>", unsafe_allow_html=True)
    difficulty = int(plan.get("difficulty_score", 5))
    confidence = int(plan.get("confidence_percentage", 70))
    st.write(f"Difficulty: **{difficulty}/10**")
    st.progress(difficulty / 10)
    st.write(f"Completion confidence: **{confidence}%**")
    st.progress(confidence / 100)
    total_days = int(plan.get("days", 1))
    finish_date = (datetime.now().date() + timedelta(days=total_days)).strftime("%d %b %Y")
    st.success(f"If you start today, you finish by: **{finish_date}**")
    tpl = plan.get("matched_template", {})
    if tpl:
        st.info(f"Matched template: {tpl.get('name', 'N/A')}")
    chips = []
    for tool in plan.get("tools", []):
        icon = TECH_ICONS.get(str(tool).lower(), "🧩")
        chips.append(f"<span class='chip'>{icon} {tool}</span>")
    chips = "".join(chips)
    st.markdown(chips, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_flowchart(plan: Dict[str, Any]):
    day_plan = plan.get("day_plan", {})

    def task_summary(day_num: int) -> str:
        tasks = day_plan.get(f"day_{day_num}", {}).get("tasks", [])
        if not tasks:
            return "Generated from timeline"
        one = tasks[0]
        if isinstance(one, dict):
            return str(one.get("summary", "Task"))
        return str(one)

    mermaid = f"""
flowchart TD
    A[Project Idea] --> B[Phase 1: Planning\\n{task_summary(1)}]
    B --> C[Phase 2: Development\\n{task_summary(2)}\\n{task_summary(3)}]
    C --> D[Phase 3: Testing\\n{task_summary(4)}]
    D --> E[Phase 4: Polish & Demo\\n{task_summary(5)}]
    E --> F[Final Deliverables]
    F --> G[Milestones Achieved]
    P[Project] --> H[High Risks]
    H --> I[Mitigation steps]

    classDef planning fill:#1e3a8a,color:#fff,stroke:#60a5fa,stroke-width:2px;
    classDef development fill:#14532d,color:#fff,stroke:#4ade80,stroke-width:2px;
    classDef testing fill:#7c2d12,color:#fff,stroke:#fb923c,stroke-width:2px;
    classDef polish fill:#4c1d95,color:#fff,stroke:#a78bfa,stroke-width:2px;
    classDef final fill:#155e75,color:#fff,stroke:#22d3ee,stroke-width:2px;

    class B planning;
    class C development;
    class D testing;
    class E polish;
    class F,G,H,I final;
"""
    components.html(
        f"""
<div style="background:#0d0d0d;border:1px solid #2a2a44;border-radius:10px;padding:10px;">
  <div class="mermaid">
{mermaid}
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true, theme:'dark'}});</script>
        """,
        height=760,
        scrolling=True,
    )
    st.caption("Quick jump to Day Plan:")
    cols = st.columns(5)
    for i in range(1, 6):
        with cols[i - 1]:
            if st.button(f"Day {i}", key=f"flow_day_{i}", use_container_width=True):
                st.session_state["current_day"] = i


def render_day_plan(plan: Dict[str, Any]):
    day_plan = plan.get("day_plan", {})
    total_days = len(day_plan)
    if total_days == 0:
        st.warning("No day-wise plan available.")
        return

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅ Previous Day", use_container_width=True):
            st.session_state["current_day"] = max(1, st.session_state["current_day"] - 1)
    with col3:
        if st.button("Next Day ➡", use_container_width=True):
            st.session_state["current_day"] = min(total_days, st.session_state["current_day"] + 1)

    selected_day = st.session_state["current_day"]
    day_key = f"day_{selected_day}"
    selected = day_plan.get(day_key, {})
    original_tasks = selected.get("tasks", [])
    normalized_tasks = [normalize_task(task) for task in original_tasks]
    redo_map = st.session_state.get("redo_tasks", {})
    day_redo_tasks = redo_map.get(selected_day, [])

    completed_days_count = 0
    for d in range(1, total_days + 1):
        day_tasks = day_plan.get(f"day_{d}", {}).get("tasks", [])
        day_redo = redo_map.get(d, [])
        _, _, is_done = day_completion_status(d, day_tasks, day_redo)
        if is_done:
            completed_days_count += 1

    progress_text = f"{completed_days_count} of {total_days} days completed"
    st.write(progress_text)
    st.progress(completed_days_count / max(total_days, 1))

    nav = []
    for d in range(1, total_days + 1):
        day_tasks = day_plan.get(f"day_{d}", {}).get("tasks", [])
        day_redo = redo_map.get(d, [])
        _, _, is_done = day_completion_status(d, day_tasks, day_redo)
        mark = "✅" if is_done else "•"
        if d == selected_day:
            nav.append(f"**{mark} Day {d}**")
        else:
            nav.append(f"{mark} Day {d}")
    st.caption(" | ".join(nav))

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(f"Day {selected_day}: {selected.get('phase', 'Execution')}")
    st.write("Original Tasks:")

    completed_tasks = st.session_state.get("completed_tasks", {})
    day_completed_indices = set(completed_tasks.get(day_key, []))

    for idx, task in enumerate(normalized_tasks):
        checkbox_key = f"task_{selected_day}_{idx}"
        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = idx in day_completed_indices

        row_col1, row_col2 = st.columns([0.12, 0.88], vertical_alignment="top")
        with row_col1:
            checked = st.checkbox("", key=checkbox_key)
        summary_text = task.get("summary", "Task")
        summary_text = f"~~{summary_text}~~" if checked else summary_text
        border_color = "#00ff88" if checked else "#00d4ff"
        with row_col2:
            st.markdown(
                (
                    f"<div style='border-left:4px solid {border_color}; padding:4px 0 4px 10px; margin-bottom:6px;'>"
                    f"{'<span style=\"text-decoration:line-through;\">' + task.get('summary', 'Task') + '</span>' if checked else task.get('summary', 'Task')}"
                    f"</div>"
                ),
                unsafe_allow_html=True,
            )
            with st.expander(f"Task {idx + 1}: {summary_text}", expanded=False):
                st.markdown(f"**📋 What to do:**  \n{task.get('what', task.get('summary', ''))}")
                how_text = task.get("how", "")
                st.markdown("**🛠️ How to do it:**")
                if any(token in how_text.lower() for token in ["mkdir", "touch", "pip ", "python ", "streamlit", "git ", "cd "]):
                    st.code(how_text, language="bash")
                else:
                    st.write(how_text or task.get("summary", ""))
                st.markdown(f"**✅ Expected Output:**  \n{task.get('output', '')}")
                st.markdown(f"**⏱️ Estimated Time:** {task.get('time', '45 mins')}")

        if checked:
            day_completed_indices.add(idx)
        elif idx in day_completed_indices:
            day_completed_indices.remove(idx)

    completed_tasks[day_key] = sorted(day_completed_indices)
    st.session_state["completed_tasks"] = completed_tasks

    st.write(f"Duration: `{selected.get('duration_estimate', '')}`")
    st.write(f"Key output: {selected.get('key_output', '')}")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Redo This Day", key=f"redo_btn_{day_key}", use_container_width=True):
        st.session_state[f"show_redo_{day_key}"] = True

    if st.session_state.get(f"show_redo_{day_key}", False):
        st.markdown("#### Select Incomplete Tasks")
        st.caption("Choose missed tasks to regenerate only for this day.")
        for idx, task in enumerate(normalized_tasks):
            st.checkbox(f"Missed: {task.get('summary', 'Task')}", key=f"redo_pick_{day_key}_{idx}")

        if st.button("Generate Revised Tasks", key=f"redo_gen_{day_key}", use_container_width=True):
            missed_tasks = [
                task.get("summary", "Task")
                for idx, task in enumerate(normalized_tasks)
                if st.session_state.get(f"redo_pick_{day_key}_{idx}", False)
            ]
            if not missed_tasks:
                st.warning("Select at least one incomplete task to regenerate.")
            else:
                revised = regenerate_missed_tasks(plan, selected_day, total_days, missed_tasks)
                redo_map[selected_day] = revised
                st.session_state["redo_tasks"] = redo_map
                for idx in range(len(revised)):
                    st.session_state[f"redo_done_{selected_day}_{idx}"] = False
                st.success("Revised tasks generated.")

    day_redo_tasks = st.session_state.get("redo_tasks", {}).get(selected_day, [])
    if day_redo_tasks:
        st.markdown("<div class='redo-card'>", unsafe_allow_html=True)
        st.markdown("#### Revised Tasks")
        for idx, item in enumerate(day_redo_tasks):
            st.markdown("##### 🔁 Revised Task")
            st.write(item.get("task", ""))
            st.caption(f"Duration: {item.get('duration', '1 hour')} | Tip: {item.get('tip', '')}")
            done_key = f"redo_done_{selected_day}_{idx}"
            if not st.session_state.get(done_key, False):
                if st.button("✅ Mark Revised Task Done", key=f"redo_done_btn_{selected_day}_{idx}"):
                    st.session_state[done_key] = True
            else:
                st.markdown("<span class='success'>✅ Revised task completed</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    day_done_count, day_total_count, is_done = day_completion_status(selected_day, normalized_tasks, day_redo_tasks)
    status = "✅ Completed" if is_done else "In Progress"
    st.caption(f"Day {selected_day} progress: {day_done_count}/{day_total_count} tasks done ({status})")
    if is_done:
        st.success(f"🎉 Day {selected_day} fully completed!")

    if st.button("Generate Standup", key=f"standup_btn_{selected_day}", use_container_width=True):
        prev_day = max(1, selected_day - 1)
        prev_tasks = plan.get("day_plan", {}).get(f"day_{prev_day}", {}).get("tasks", [])
        prev_summary = ", ".join(
            [normalize_task(t).get("summary", "task") for t in prev_tasks[:2]]
        ) or "No updates"
        today_summary = ", ".join([task.get("summary", "task") for task in normalized_tasks[:3]]) or "No tasks listed"
        blockers = ", ".join(
            [(r.get("risk", "") if isinstance(r, dict) else str(r)) for r in plan.get("risks", [])[:2]]
        ) or "No blockers"
        st.session_state[f"standup_{selected_day}"] = (
            f"Yesterday: {prev_summary}\n"
            f"Today: {today_summary}\n"
            f"Blockers: {blockers}"
        )

    if st.session_state.get(f"standup_{selected_day}"):
        st.text_area(
            "Standup Message (copy-ready)",
            value=st.session_state.get(f"standup_{selected_day}", ""),
            height=120,
            key=f"standup_text_{selected_day}",
        )


def render_features(plan: Dict[str, Any]):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### MVP Features")
        for feature in plan.get("mvp_features", []):
            st.markdown(f"<span class='success'>✅ {feature}</span>", unsafe_allow_html=True)
    with c2:
        st.markdown("#### Optional Features")
        for feature in plan.get("optional_features", []):
            st.markdown(f"<span class='warn'>⭐ {feature}</span>", unsafe_allow_html=True)

    st.markdown("#### Final Deliverables")
    for deliverable in plan.get("deliverables", []):
        st.write(f"- {deliverable}")


def render_risks_and_milestones(plan: Dict[str, Any]):
    risks = plan.get("risks", [])
    st.markdown("#### Risk Register")
    if not risks:
        st.info("No risks found.")
    else:
        high, medium, low = [], [], []
        for item in risks:
            if isinstance(item, dict):
                sev = str(item.get("severity", "Medium")).title()
                risk_text = item.get("risk", "")
            else:
                sev = "Medium"
                risk_text = str(item)
            if sev == "High":
                high.append(risk_text)
            elif sev == "Low":
                low.append(risk_text)
            else:
                medium.append(risk_text)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**High Severity**")
            for risk in high:
                st.markdown(f"<div class='card'><span class='danger'>●</span> {risk}</div>", unsafe_allow_html=True)
            if not high:
                st.caption("No high risks")
        with c2:
            st.markdown("**Medium Severity**")
            for risk in medium:
                st.markdown(f"<div class='card'><span class='warn'>●</span> {risk}</div>", unsafe_allow_html=True)
            if not medium:
                st.caption("No medium risks")
        with c3:
            st.markdown("**Low Severity**")
            for risk in low:
                st.markdown(f"<div class='card'><span class='success'>●</span> {risk}</div>", unsafe_allow_html=True)
            if not low:
                st.caption("No low risks")

    st.markdown("#### Milestone Timeline")
    milestones = plan.get("milestones", [])
    if not milestones:
        st.info("No milestones found.")
        return
    for idx, mile in enumerate(milestones, start=1):
        st.markdown(
            f"<div class='card'><b>Milestone {idx}</b><br/>{mile}</div>",
            unsafe_allow_html=True,
        )


def render_export(plan: Dict[str, Any]):
    st.markdown("#### Download")
    plan_json = json.dumps(plan, indent=2)
    st.download_button("Download as JSON", plan_json, file_name="project_plan.json", mime="application/json")

    pdf_exporter = PDFExporter()
    pdf_path = BASE_DIR / "saved_plans" / "latest_plan.pdf"
    pdf_exporter.build_pdf(plan, pdf_path)
    st.download_button("Download as PDF", data=pdf_path.read_bytes(), file_name="project_plan.pdf", mime="application/pdf")

    if st.button("Save plan locally", use_container_width=True):
        saved_path = save_plan_locally(plan, BASE_DIR)
        st.success(f"Saved to {saved_path.name}")

    st.markdown("#### Previously Saved Plans")
    for p in list_saved_plans(BASE_DIR)[:15]:
        col1, col2 = st.columns([3, 1])
        col1.write(p.name)
        if col2.button("Load", key=f"load_{p.name}"):
            load_selected_plan(p)


def main():
    st.set_page_config(page_title="AI Project Builder Agent", page_icon="🤖", layout="wide")
    inject_styles()
    init_state()

    with st.sidebar:
        st.title("🤖 AI Project Builder")
        st.caption("Professional Roadmap Generator")
        provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
        project_idea = st.text_area(
            "Project Idea",
            value="",
            placeholder="e.g. AI chatbot for student queries",
        )
        extra_features = st.text_area(
            "Extra Features / Requirements (optional)",
            value="",
            placeholder="e.g. Voice input, admin dashboard, PDF export, deployment on Render, multi-language support",
        )
        days_input = st.text_input("Timeline (days)", value="", placeholder="e.g. 14")
        skill_level = st.selectbox(
            "Skill Level",
            ["Beginner", "Intermediate", "Advanced"],
            index=None,
            placeholder="Select skill level",
        )
        team_size_input = st.text_input("Team Size", value="", placeholder="e.g. 3")
        daily_hours_input = st.text_input("Daily Hours", value="", placeholder="e.g. 8")
        if st.button("Generate Plan", use_container_width=True):
            if not project_idea.strip():
                st.warning("Please enter a project idea.")
                st.stop()
            if not skill_level:
                st.warning("Please select a skill level.")
                st.stop()
            try:
                days = max(1, min(90, int(days_input.strip())))
                team_size = max(1, min(10, int(team_size_input.strip())))
                daily_hours = max(1, min(12, int(daily_hours_input.strip())))
            except Exception:
                st.warning("Enter valid numeric values for days, team size, and daily hours.")
                st.stop()
            with st.spinner("Building your professional roadmap..."):
                st.session_state["plan_result"] = run_agent(
                    project_idea=project_idea.strip(),
                    extra_features=extra_features.strip(),
                    days=days,
                    skill_level=skill_level,
                    team_size=team_size,
                    daily_hours=daily_hours,
                    provider=provider,
                )
                st.session_state["current_day"] = 1
                st.session_state["redo_tasks"] = {}
                st.session_state["completed_tasks"] = {}

        st.markdown("---")
        st.markdown("### Saved Plans")
        saved = list_saved_plans(BASE_DIR)
        selected = st.selectbox("Select saved plan", ["None"] + [s.name for s in saved], index=0)
        if selected != "None" and st.button("Reload Selected Plan", use_container_width=True):
            match = next((s for s in saved if s.name == selected), None)
            if match:
                load_selected_plan(match)

    plan = st.session_state.get("plan_result")
    if not plan:
        st.info("Fill the sidebar and click Generate Plan.")
        return

    tabs = st.tabs(["Overview", "Day-wise Plan", "Features", "Risks & Milestones", "Flowchart", "Export"])
    with tabs[0]:
        render_overview(plan)
    with tabs[1]:
        render_day_plan(plan)
    with tabs[2]:
        render_features(plan)
    with tabs[3]:
        render_risks_and_milestones(plan)
    with tabs[4]:
        render_flowchart(plan)
    with tabs[5]:
        render_export(plan)


if __name__ == "__main__":
    main()
