import io
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import pandas as pd
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
    theme = st.session_state.get("theme", "dark")
    if theme == "dark":
        bg_color = "#0d0d0d"
        card_bg = "#1a1a2e"
        text_color = "#f5f5f5"
        accent = "#00d4ff"
        border = "#2a2a44"
        secondary_bg = "#132838"
        chip_bg = "#102234"
        chip_border = "#1d4d69"
        expander_bg = "#141428"
        expander_open = "#1b1b32"
    else:
        bg_color = "#f5f7fa"
        card_bg = "#ffffff"
        text_color = "#1a1a2e"
        accent = "#0066cc"
        border = "#e0e0e0"
        secondary_bg = "#f0f2f5"
        chip_bg = "#e8f4fd"
        chip_border = "#b3d9ff"
        expander_bg = "#ffffff"
        expander_open = "#f8f9fa"
    
    st.markdown(
        f"""
        <style>
        :root {{
            --bg-color: {bg_color};
            --card-bg: {card_bg};
            --text-color: {text_color};
            --accent: {accent};
            --border: {border};
            --secondary-bg: {secondary_bg};
            --chip-bg: {chip_bg};
            --chip-border: {chip_border};
            --expander-bg: {expander_bg};
            --expander-open: {expander_open};
        }}
        .stApp {{ background-color: var(--bg-color); color: var(--text-color); font-family: "Inter", "Segoe UI", sans-serif; }}
        .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 14px; padding: 14px; margin: 8px 0; transition: 0.25s; }}
        .card:hover {{ box-shadow: 0 0 12px rgba(0,212,255,0.28); border-color: var(--accent); }}
        .badge {{ display:inline-block; padding:4px 10px; border-radius: 999px; background:var(--secondary-bg); color:var(--accent); border:1px solid var(--accent); font-size:12px; margin:4px 6px 4px 0; }}
        .chip {{ display:inline-block; padding:5px 10px; border-radius: 999px; background:var(--chip-bg); color:var(--text-color); border:1px solid var(--chip-border); margin:4px 6px 4px 0; }}
        .success {{ color:#00ff88; }}
        .warn {{ color:#ffaa00; }}
        .danger {{ color:#ff4466; }}
        .mono {{ font-family: "Consolas", "Courier New", monospace; }}
        .tiny {{ font-size: 12px; opacity: 0.85; }}
        .redo-card {{ background: #231b12; border: 1px solid #ffaa00; border-radius: 14px; padding: 14px; margin: 8px 0; }}
        div[data-testid="stExpander"] {{ background: var(--expander-bg); border: 1px solid var(--border); border-radius: 10px; margin-bottom: 8px; }}
        div[data-testid="stExpander"] details[open] {{ background: var(--expander-open); }}
        div[data-testid="stProgressBar"] > div > div > div > div {{ background-color: var(--accent); transition: all 0.4s ease; }}
        .pulse {{ animation: pulse 2s infinite; }}
        @keyframes pulse {{ 0% {{ transform: scale(1); }} 50% {{ transform: scale(1.05); }} 100% {{ transform: scale(1); }} }}
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
        "current_day": 1,
        "redo_tasks": {},
        "completed_tasks": {},
        "theme": "dark",  # Feature 14: Dark/Light mode
        "streak": 0,  # Feature 9: Streak tracker
        "best_streak": 0,
        "assignments": {},  # Feature 7: Team task assigner # Feature 16: Project thumbnail
        "timer_running": False,  # Feature 8: Daily timer
        "current_timer_task": None,
        "completion_history": [],  # Feature 13: Auto difficulty adjuster
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


def clean_project_title(title: str) -> str:
    import re
    if not title:
        return "Project"
    cleaned = re.sub(r'^[\W_]+', '', str(title).strip())
    return cleaned or title


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
        st.session_state["project_idea"] = payload.get("project_idea", "")
        st.session_state["current_day"] = 1
        st.session_state["redo_tasks"] = {}
        st.session_state["completed_tasks"] = {}
        st.success(f"Loaded {file_path.name}")
    except Exception as exc:
        st.error(f"Could not load plan: {exc}")


def render_overview(plan: Dict[str, Any]):
    project_name = st.session_state.get("project_idea", "")
    if project_name:
        st.markdown(f"""
            <div style="text-align: center; padding: 20px 0;">
                <h1 style="font-size: 2.5rem; font-weight: 800; color: white; margin: 0;">
                    {project_name}
                </h1>
            </div>
        """, unsafe_allow_html=True)
    else:
        return

    category = plan.get('category', 'General')
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<span class='badge'>{category}</span>", unsafe_allow_html=True)
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

    # Feature 11: Plan Optimizer
    if st.button("🔧 Optimize My Plan", use_container_width=True):
        optimize_plan(plan)

    # Feature 13: Auto Difficulty Adjuster
    check_auto_difficulty_adjustment(plan)


def draw_project_phases_flowchart(day_plan, milestones):
    fig, ax = plt.subplots(figsize=(12, 16))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    ax.set_xlim(0, 12)
    ax.set_ylim(-1, 16)  # Add padding at bottom
    ax.axis('off')
    
    # Colors
    colors = {
        'project_idea': '#1a1a2e',
        'planning': '#4A90D9',
        'development': '#27AE60', 
        'testing': '#E67E22',
        'polish': '#8E44AD',
        'deliverables': '#16A085',
        'milestones': '#2C3E50'
    }
    
    # Positions - better spacing
    y_positions = [14, 11.5, 9, 6.5, 4, 1.5, -0.5]  # More space between boxes
    x_center = 6
    
    # Nodes
    nodes = [
        ("Project Idea", 'project_idea', 14),
        ("Day 1 - Planning", 'planning', 11.5),
        ("Day 2 - Development", 'development', 9),
        ("Day 3 - Development", 'development', 6.5),
        ("Day 4 - Testing", 'testing', 4),
        ("Day 5 - Polish & Demo", 'polish', 1.5),
        ("Final Deliverables", 'deliverables', -0.5)
    ]
    
    # Draw nodes
    for i, (text, color_key, y) in enumerate(nodes):
        color = colors.get(color_key, '#333333')
        # Box - wider and taller
        box = FancyBboxPatch((x_center-3, y-0.4), 6, 0.8, 
                           boxstyle="round,pad=0.2", 
                           facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(box)
        
        # Phase name in bold white text
        ax.text(x_center, y+0.1, text, ha='center', va='center', 
               fontsize=13, fontweight='bold', color='white')
        
        # Tasks for days
        if i > 0 and i < 6:  # Days 1-5
            tasks = day_plan.get(f"day_{i}", {}).get("tasks", [])
            if tasks:
                task_summaries = [t.get("summary", str(t)) for t in tasks[:2]]
                task_text = "\n".join([s[:45] + "..." if len(s) > 45 else s for s in task_summaries])
                if len(tasks) > 2:
                    task_text += f"\n+ {len(tasks)-2} more tasks"
                ax.text(x_center, y-0.6, task_text, ha='center', va='top', 
                       fontsize=8, color='lightgray', linespacing=0.9)
        
        # Arrows
        if i < len(nodes)-1:
            ax.annotate('', xy=(x_center, y-0.4), xytext=(x_center, nodes[i+1][2]+0.4),
                       arrowprops=dict(arrowstyle='->', color='white', lw=3, shrinkA=0, shrinkB=0))
    
    # Milestones at bottom with better styling
    if milestones:
        milestone_text = "Milestones:\n" + "\n".join(milestones[:5])  # Limit to 5
        if len(milestones) > 5:
            milestone_text += f"\n... and {len(milestones)-5} more"
        
        # Milestone box
        ms_box = FancyBboxPatch((x_center-3, -2.5), 6, 1.5, 
                               boxstyle="round,pad=0.3", 
                               facecolor=colors['milestones'], edgecolor='white', linewidth=2)
        ax.add_patch(ms_box)
        ax.text(x_center, -2, milestone_text, ha='center', va='center', 
               fontsize=10, color='white', linespacing=1.2)
    
    plt.tight_layout(pad=2.0)
    return fig


def draw_risk_flowchart(risks):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Group risks by severity
    high_risks = [r['risk'] for r in risks if r.get('severity') == 'High']
    medium_risks = [r['risk'] for r in risks if r.get('severity') == 'Medium'] 
    low_risks = [r['risk'] for r in risks if r.get('severity') == 'Low']
    
    # Colors
    colors = {'high': '#E74C3C', 'medium': '#F39C12', 'low': '#27AE60'}
    
    # Nodes
    nodes = [
        ("Project", None, 10),
        (f"High Risks ({len(high_risks)})", 'high', 7),
        (f"Medium Risks ({len(medium_risks)})", 'medium', 4),
        (f"Low Risks ({len(low_risks)})", 'low', 1)
    ]
    
    x_center = 6
    
    # Draw nodes
    for i, (text, color_key, y) in enumerate(nodes):
        color = colors.get(color_key, '#333333')
        # Box
        box = FancyBboxPatch((x_center-3, y-0.5), 6, 1, 
                           boxstyle="round,pad=0.1", 
                           facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(box)
        
        # Text
        ax.text(x_center, y, text, ha='center', va='center', 
               fontsize=12, fontweight='bold', color='white')
        
        # Risk lists
        if color_key == 'high' and high_risks:
            risk_text = "\n".join([r[:40] + "..." if len(r) > 40 else r for r in high_risks[:3]])  # Limit to 3, shorten
            if len(high_risks) > 3:
                risk_text += f"\n... and {len(high_risks)-3} more"
            ax.text(x_center, y-0.8, risk_text, ha='center', va='top', 
                   fontsize=7, color='white')
        elif color_key == 'medium' and medium_risks:
            risk_text = "\n".join([r[:40] + "..." if len(r) > 40 else r for r in medium_risks[:3]])
            if len(medium_risks) > 3:
                risk_text += f"\n... and {len(medium_risks)-3} more"
            ax.text(x_center, y-0.8, risk_text, ha='center', va='top', 
                   fontsize=7, color='white')
        elif color_key == 'low' and low_risks:
            risk_text = "\n".join([r[:40] + "..." if len(r) > 40 else r for r in low_risks[:3]])
            if len(low_risks) > 3:
                risk_text += f"\n... and {len(low_risks)-3} more"
            ax.text(x_center, y-0.8, risk_text, ha='center', va='top', 
                   fontsize=7, color='white')
        
        # Arrows
        if i < len(nodes)-1:
            ax.annotate('', xy=(x_center, y-0.5), xytext=(x_center, nodes[i+1][2]+0.5),
                       arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    return fig


def render_flowchart(plan: Dict[str, Any]):
    day_plan = plan.get("day_plan", {})
    risks = plan.get("risks", [])
    milestones = plan.get("milestones", [])

    # Project Phases Flowchart
    st.subheader("Project Phases Flowchart")
    fig1 = draw_project_phases_flowchart(day_plan, milestones)
    st.pyplot(fig1)
    
    # Download button for phases flowchart
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=150, bbox_inches='tight')
    buf1.seek(0)
    st.download_button(
        label="Download Project Phases Flowchart as PNG",
        data=buf1.getvalue(),
        file_name="project_phases_flowchart.png",
        mime="image/png",
        key="download_phases"
    )
    plt.close(fig1)

    st.markdown("---")

    # Risk Flowchart
    st.subheader("Risk Assessment Flowchart")
    fig2 = draw_risk_flowchart(risks)
    st.pyplot(fig2)
    
    # Download button for risk flowchart
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
    buf2.seek(0)
    st.download_button(
        label="Download Risk Flowchart as PNG",
        data=buf2.getvalue(),
        file_name="risk_flowchart.png",
        mime="image/png",
        key="download_risks"
    )
    plt.close(fig2)

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

    # Feature 8: Display active timer
    display_active_timer()

    completed_days_count = 0

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
                
                # Feature 1: AI Code Snippet Generator
                if st.button("💻 Get Starter Code", key=f"code_{selected_day}_{idx}"):
                    generate_code_snippet(task, plan)
                
                # Feature 8: Daily Timer
                timer_key = f"timer_{selected_day}_{idx}"
                if st.button("⏰ Start Timer", key=f"start_timer_{selected_day}_{idx}"):
                    start_task_timer(task, timer_key)
                
                # Feature 12: Stuck? Get Help
                if st.button("😕 I'm Stuck", key=f"stuck_{selected_day}_{idx}"):
                    get_stuck_help(task, plan)

        if checked:
            day_completed_indices.add(idx)
        elif idx in day_completed_indices:
            day_completed_indices.remove(idx)

    completed_tasks[day_key] = sorted(day_completed_indices)
    st.session_state["completed_tasks"] = completed_tasks

    # Feature 15: Check for project completion
    check_project_completion(plan)

    st.write(f"Duration: `{selected.get('duration_estimate', '')}`")
    st.write(f"Key output: {selected.get('key_output', '')}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Feature 7: Team Task Assigner
    team_size = plan.get("team_size", 1)
    if team_size > 1:
        if st.button("👥 Assign Tasks to Team", key=f"assign_{selected_day}", use_container_width=True):
            assign_team_tasks(selected_day, normalized_tasks, team_size)

    # Feature 9: Update streak tracker
    update_streak_tracker(plan)

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



def render_features(plan: Dict[str, Any]):
    st.markdown("### Project Feature Summary")
    st.markdown("Review the scope, deliverables, and execution highlights for your plan.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### MVP Features")
        mvp_features = plan.get("mvp_features", [])
        if mvp_features:
            for feature in mvp_features:
                st.markdown(
                    f"<div class='card'><span class='success'>✅</span> {feature}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No MVP features available.")
    with c2:
        st.markdown("#### Optional Features")
        optional_features = plan.get("optional_features", [])
        if optional_features:
            for feature in optional_features:
                st.markdown(
                    f"<div class='card'><span class='warn'>⭐</span> {feature}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No optional features suggested.")

    st.markdown("#### Final Deliverables")
    deliverables = plan.get("deliverables", [])
    if deliverables:
        for deliverable in deliverables:
            st.markdown(f"<div class='card'>{deliverable}</div>", unsafe_allow_html=True)
    else:
        st.info("No deliverables defined yet.")


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
    st.set_page_config(page_title="Project Builder", page_icon="📘", layout="wide")
    inject_styles()
    init_state()

    with st.sidebar:
        st.title("Project Builder")
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
                st.session_state["project_idea"] = project_idea.strip()
                st.session_state["current_day"] = 1
                st.session_state["redo_tasks"] = {}
                st.session_state["completed_tasks"] = {}

        st.markdown("---")
        st.markdown("### Saved Plans")
        saved = list_saved_plans(BASE_DIR)
        saved_names = [s.name for s in saved]
        selected = st.selectbox("Select saved plan", ["None"] + saved_names, index=0)
        if selected != "None" and st.button("Reload Selected Plan", use_container_width=True):
            match = next((s for s in saved if s.name == selected), None)
            if match:
                load_selected_plan(match)

        # Feature 9: Streak Tracker
        st.markdown("---")
        streak = st.session_state.get("streak", 0)
        best_streak = st.session_state.get("best_streak", 0)
        if streak > 0:
            st.markdown(f"🔥 **{streak} Day Streak!**")
            if streak >= 6:
                st.markdown("*Unstoppable! 🚀*")
            elif streak >= 3:
                st.markdown("*You're on fire! 🔥*")
            else:
                st.markdown("*Good start! Keep going 💪*")
        else:
            st.markdown("🔥 **0 Day Streak**")
        if best_streak > 0:
            st.markdown(f"🏆 Best: {best_streak} days")

    plan = st.session_state.get("plan_result")
    if not plan:
        st.info("Fill the sidebar and click Generate Plan.")
        return

    tabs = st.tabs(["Overview", "Day-wise Plan", "Features", "Risks & Milestones", "Export", "Compare Projects", "README Generator"] )
    
    # Feature 14: Dark/Light Mode Toggle
    col1, col2 = st.columns([11, 1])
    with col2:
        theme_icon = "🌙" if st.session_state["theme"] == "dark" else "☀️"
        if st.button(theme_icon, key="theme_toggle", help="Toggle theme"):
            st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
            st.rerun()
    
    with tabs[0]:
        render_overview(plan)
    with tabs[1]:
        render_day_plan(plan)
    with tabs[2]:
        render_features(plan)
    with tabs[3]:
        render_risks_and_milestones(plan)
    with tabs[4]:
        render_export(plan)
    with tabs[5]:
        render_compare_projects()
    with tabs[6]:
        render_readme_generator(plan)


# Feature 6: Burndown Chart
def create_burndown_chart(plan):
    day_plan = plan.get("day_plan", {})
    completed_tasks = st.session_state.get("completed_tasks", {})
    
    total_days = len(day_plan)
    if total_days == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No day plan available", ha='center', va='center')
        return fig
    
    days = list(range(1, total_days + 1))
    ideal_remaining = []
    actual_remaining = []
    
    total_tasks = 0
    for day in days:
        day_key = f"day_{day}"
        if day_key in day_plan:
            tasks = day_plan[day_key].get("tasks", [])
            total_tasks += len(tasks)
    
    remaining = total_tasks
    for day in days:
        # Ideal: linear decrease
        ideal_remaining.append(remaining - (total_tasks / total_days) * (day - 1))
        
        # Actual: based on completed tasks
        day_completed = len(completed_tasks.get(f"day_{day}", []))
        remaining -= day_completed
        actual_remaining.append(max(0, remaining))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(days, ideal_remaining, 'k--', label='Ideal Burndown', linewidth=2)
    ax.plot(days, actual_remaining, 'c-', label='Actual Burndown', linewidth=3)
    ax.fill_between(days, ideal_remaining, actual_remaining, 
                   where=[a > i for a, i in zip(actual_remaining, ideal_remaining)], 
                   color='red', alpha=0.3, label='Behind Schedule')
    ax.fill_between(days, ideal_remaining, actual_remaining, 
                   where=[a < i for a, i in zip(actual_remaining, ideal_remaining)], 
                   color='green', alpha=0.3, label='Ahead of Schedule')
    
    ax.set_xlabel('Day')
    ax.set_ylabel('Remaining Tasks')
    ax.set_title('Project Burndown Chart')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Calculate status
    if actual_remaining and ideal_remaining:
        diff = actual_remaining[-1] - ideal_remaining[-1]
        if diff > 0:
            status = f"You are {int(diff)} tasks behind schedule"
        elif diff < 0:
            status = f"You are {int(abs(diff))} tasks ahead of schedule"
        else:
            status = "You are on track!"
        st.caption(status)
    
    return fig


# Feature 11: Plan Optimizer
def optimize_plan(plan):
    client = LLMClient(plan.get("provider", "groq"))
    if not client.is_configured():
        st.error("LLM not configured for plan optimization")
        return

    day_plan = plan.get("day_plan", {})
    day_plan_summary = []
    for day_key in sorted(day_plan.keys(), key=lambda k: int(k.split("_")[1]) if "_" in k else 0):
        day = day_plan.get(day_key, {})
        tasks = day.get("tasks", [])
        task_names = []
        for task in tasks[:3]:
            if isinstance(task, dict):
                task_names.append(task.get("summary", str(task)))
            else:
                task_names.append(str(task))
        day_plan_summary.append(f"{day_key}: {len(tasks)} tasks - {', '.join(task_names) if task_names else 'No tasks'}")
    day_plan_summary = "\n".join(day_plan_summary) or "No day-wise plan available."

    prompt = f"""
You are a senior project manager reviewing this project plan.

Project: {plan.get('project_idea', '')}
Total days: {plan.get('days', 5)}
Skill level: {plan.get('skill_level', 'Beginner')}
Daily hours: {plan.get('daily_hours', plan.get('daily_hours', 8))}

Current day-wise plan:
{day_plan_summary}

Review this plan and provide:
1. STRENGTHS: 2-3 things the plan does well
2. PROBLEMS: specific days that are overloaded or underloaded
3. SUGGESTIONS: 3-4 concrete improvements
4. REBALANCED PLAN: suggest which tasks to move between days

Return as JSON:
{{
  "strengths": ["point1", "point2"],
  "problems": ["problem1", "problem2"],
  "suggestions": ["suggestion1", "suggestion2"],
  "rebalanced": {{
    "day_1": "move task X to day 2",
    "day_3": "add more testing tasks"
  }}
}}
"""

    try:
        response = client.call_llm(prompt=prompt, system_prompt="You are a senior project manager reviewing project plans.")
        parsed = {}
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            parsed = safe_json_loads(response[json_start:json_end], {})
        if not isinstance(parsed, dict):
            parsed = safe_json_loads(response, {})

        strengths = parsed.get("strengths", []) if isinstance(parsed.get("strengths", []), list) else []
        problems = parsed.get("problems", []) if isinstance(parsed.get("problems", []), list) else []
        suggestions = parsed.get("suggestions", []) if isinstance(parsed.get("suggestions", []), list) else []
        rebalanced = parsed.get("rebalanced", {}) if isinstance(parsed.get("rebalanced", {}), dict) else {}

        st.subheader("🔧 Plan Optimization Report")
        if strengths:
            st.markdown("<div style='background:#163f1f; color:#d4ffd4; padding:14px; border-radius:12px; margin-bottom:10px;'><strong>✅ Strengths</strong><br/>" + "<br/>".join(str(s) for s in strengths) + "</div>", unsafe_allow_html=True)
        if problems:
            st.markdown("<div style='background:#4a2f00; color:#ffe8a1; padding:14px; border-radius:12px; margin-bottom:10px;'><strong>⚠️ Problems</strong><br/>" + "<br/>".join(str(p) for p in problems) + "</div>", unsafe_allow_html=True)
        if suggestions:
            st.markdown("<div style='background:#102a4d; color:#cfe8ff; padding:14px; border-radius:12px; margin-bottom:10px;'><strong>💡 Suggestions</strong><br/>" + "<br/>".join(str(s) for s in suggestions) + "</div>", unsafe_allow_html=True)
        if rebalanced:
            rebalanced_lines = [f"{day}: {move}" for day, move in rebalanced.items()]
            st.markdown("<div style='background:#2f1e3b; color:#e7d6ff; padding:14px; border-radius:12px; margin-bottom:10px;'><strong>🔄 Rebalanced Plan</strong><br/>" + "<br/>".join(rebalanced_lines) + "</div>", unsafe_allow_html=True)

        if not (strengths or problems or suggestions or rebalanced):
            st.warning("The plan optimizer returned no structured output. Here is the raw response:")
            st.markdown(response)
    except Exception as e:
        st.error(f"Failed to optimize plan: {e}")


# Feature 13: Auto Difficulty Adjuster
def check_auto_difficulty_adjustment(plan):
    completion_history = st.session_state.get("completion_history", [])
    if len(completion_history) < 2:
        return
    
    recent_completion = completion_history[-2:]  # Last 2 days
    low_completion_count = sum(1 for comp in recent_completion if comp < 0.5)  # Less than 50%
    
    if low_completion_count >= 2:
        st.warning("⚠️ Looks like you're finding this tough. Want me to simplify the remaining days?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, simplify plan", use_container_width=True):
                simplify_plan(plan)
        with col2:
            if st.button("No, I'll push through", use_container_width=True):
                st.info("Keep going! You've got this! 💪")


def simplify_plan(plan):
    client = LLMClient("groq")
    if not client.is_configured():
        st.error("LLM not configured for plan simplification")
        return
    
    prompt = f"""
Simplify this project plan for a struggling developer:
Current plan: {json.dumps(plan, indent=2)}

Reduce task complexity, break big tasks into smaller ones, extend timeline by 1-2 days if needed.
Adjust difficulty score down.

Return simplified JSON plan.
"""
    
    try:
        response = client.call_llm(prompt=prompt, system_prompt="Simplify project plans for beginners.")
        simplified_plan = safe_json_loads(response, plan)
        
        plan.update(simplified_plan)
        st.session_state['plan_result'] = plan
        st.success("Plan simplified! Check the updated day plan.")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to simplify plan: {e}")


# Feature 2: Project Comparison
def render_compare_projects():
    st.header("🔍 Compare Projects")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Project 1")
        proj1_idea = st.text_area("Project Idea 1", key="proj1_idea", height=100)
        proj1_extra = st.text_area("Extra Features 1", key="proj1_extra", height=50)
        proj1_days = st.slider("Days 1", 1, 30, 7, key="proj1_days")
        proj1_skill = st.selectbox("Skill Level 1", ["Beginner", "Intermediate", "Advanced"], key="proj1_skill")
        proj1_team = st.slider("Team Size 1", 1, 5, 1, key="proj1_team")
    
    with col2:
        st.subheader("Project 2")
        proj2_idea = st.text_area("Project Idea 2", key="proj2_idea", height=100)
        proj2_extra = st.text_area("Extra Features 2", key="proj2_extra", height=50)
        proj2_days = st.slider("Days 2", 1, 30, 7, key="proj2_days")
        proj2_skill = st.selectbox("Skill Level 2", ["Beginner", "Intermediate", "Advanced"], key="proj2_skill")
        proj2_team = st.slider("Team Size 2", 1, 5, 1, key="proj2_team")
    
    if st.button("⚖️ Compare Projects", use_container_width=True):
        if not proj1_idea or not proj2_idea:
            st.error("Please enter both project ideas")
            return
        
        with st.spinner("Analyzing projects..."):
            plan1 = run_agent(proj1_idea, proj1_extra, proj1_days, proj1_skill, proj1_team, 8, "groq")
            plan2 = run_agent(proj2_idea, proj2_extra, proj2_days, proj2_skill, proj2_team, 8, "groq")
            
            # Create comparison table
            data = {
                "Feature": ["Category", "Difficulty", "Recommended Days", "Tools", "Team Size", "Best For", "Risk Level"],
                "Project 1": [
                    plan1.get("category", "N/A"),
                    f"{plan1.get('difficulty_score', 5)}/10",
                    plan1.get("days", "N/A"),
                    ", ".join(plan1.get("tools", [])[:3]) + ("..." if len(plan1.get("tools", [])) > 3 else ""),
                    plan1.get("team_size", 1),
                    "Final year project" if plan1.get("difficulty_score", 5) > 7 else "Hackathon" if plan1.get("days", 7) <= 3 else "Portfolio project",
                    "High" if any(r.get("severity") == "High" for r in plan1.get("risks", [])) else "Medium"
                ],
                "Project 2": [
                    plan2.get("category", "N/A"),
                    f"{plan2.get('difficulty_score', 5)}/10",
                    plan2.get("days", "N/A"),
                    ", ".join(plan2.get("tools", [])[:3]) + ("..." if len(plan2.get("tools", [])) > 3 else ""),
                    plan2.get("team_size", 1),
                    "Final year project" if plan2.get("difficulty_score", 5) > 7 else "Hackathon" if plan2.get("days", 7) <= 3 else "Portfolio project",
                    "High" if any(r.get("severity") == "High" for r in plan2.get("risks", [])) else "Medium"
                ]
            }
            
            df = pd.DataFrame(data)
            st.table(df)
            
            # Winner badge
            diff1 = plan1.get("difficulty_score", 5) + plan1.get("days", 7)
            diff2 = plan2.get("difficulty_score", 5) + plan2.get("days", 7)
            
            if diff1 < diff2:
                st.success("🏆 Project 1 is easier/faster!")
            elif diff2 < diff1:
                st.success("🏆 Project 2 is easier/faster!")
            else:
                st.info("🤝 Both projects are equally challenging!")


# Feature 4: GitHub README Generator
def render_readme_generator(plan):
    st.header("📖 README Generator")
    
    readme_content = f"""# {plan.get('project_idea', 'Project Name')}

## About
{plan.get('project_idea', '')}

## Features
{chr(10).join('- ' + str(f) for f in plan.get('deliverables', []))}

## Tech Stack
{chr(10).join('- ' + str(t) for t in plan.get('tools', []))}

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Run the application:
```bash
streamlit run app.py
```

## Project Structure
- `app.py` - Main application
- `agent.py` - AI agent logic
- `data/` - Project data and templates
- `utils.py` - Utility functions

## Team
- Team Size: {plan.get('team_size', 1)} members
- Skill Level: {plan.get('skill_level', 'Intermediate')}

## License
MIT License
"""
    
    st.markdown("### Preview")
    st.markdown(readme_content)
    
    if st.button("💾 Download README.md", use_container_width=True):
        st.download_button(
            label="Download README.md",
            data=readme_content,
            file_name="README.md",
            mime="text/markdown",
            key="download_readme"
        )


# Feature 1: AI Code Snippet Generator
def generate_code_snippet(task, plan):
    client = LLMClient("groq")
    if not client.is_configured():
        st.error("LLM not configured for code generation")
        return
    
    prompt = f"""
Project: {plan.get('project_idea', '')}
Task: {task.get('summary', '')}
Tech stack: {', '.join(plan.get('tools', []))}
Skill level: {plan.get('skill_level', 'Intermediate')}

Generate starter code for this specific task.
Include comments explaining each line.
Keep it simple and runnable.
Return only the code block with language specified.

Example format:
```python
# This is a comment
def hello_world():
    print("Hello, World!")  # Print greeting
```
"""
    
    try:
        response = client.call_llm(prompt=prompt, system_prompt="Generate clean, commented starter code.")
        # Extract code blocks
        import re
        code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', response, re.DOTALL)
        
        if code_blocks:
            for lang, code in code_blocks:
                st.code(code.strip(), language=lang or "python")
                if st.button("📋 Copy Code", key=f"copy_code_{hash(code)}"):
                    st.text_area("Code copied to clipboard:", value=code.strip(), height=200)
        else:
            st.code(response.strip(), language="python")
            
    except Exception as e:
        st.error(f"Failed to generate code: {e}")


# Feature 7: Team Task Assigner
def assign_team_tasks(day_num, tasks, team_size):
    if not tasks:
        st.error("No tasks to assign")
        return
    
    # Simple assignment: distribute evenly
    assignments = {f"Member {i+1}": [] for i in range(team_size)}
    for i, task in enumerate(tasks):
        member = f"Member {(i % team_size) + 1}"
        assignments[member].append(task.get("summary", f"Task {i+1}"))
    
    st.subheader(f"👥 Day {day_num} Task Assignment")
    assignment_text = ""
    for member, member_tasks in assignments.items():
        task_list = "\n".join(f"• {task}" for task in member_tasks)
        st.markdown(f"**{member}:**\n{task_list}")
        assignment_text += f"{member}:\n{task_list}\n\n"
    
    if st.button("📋 Copy Assignment", key=f"copy_assign_{day_num}"):
        st.text_area("Team assignment:", value=assignment_text, height=200)
    
    # Store assignments
    st.session_state["assignments"][day_num] = assignments


# Feature 8: Daily Timer
def start_task_timer(task, timer_key):
    time_str = task.get("time", "45 mins")
    # Parse time (simple parsing)
    if "hour" in time_str.lower():
        hours = int(time_str.split()[0])
        total_seconds = hours * 3600
    else:
        minutes = int(time_str.split()[0])
        total_seconds = minutes * 60
    
    st.session_state["timer_running"] = True
    st.session_state["current_timer_task"] = task.get("summary", "Task")
    st.session_state["timer_end_time"] = time.time() + total_seconds
    st.session_state["timer_total"] = total_seconds


def display_active_timer():
    if st.session_state.get("timer_running", False):
        remaining = max(0, st.session_state["timer_end_time"] - time.time())
        total = st.session_state["timer_total"]
        progress = 1 - (remaining / total)
        
        # Color based on time left
        if remaining / total > 0.5:
            color = "green"
        elif remaining / total > 0.1:
            color = "orange"
        else:
            color = "red"
        
        mins, secs = divmod(int(remaining), 60)
        time_str = f"{mins:02d}:{secs:02d}"
        
        st.markdown(f"""
        <div style="background: {color}; color: white; padding: 10px; border-radius: 10px; text-align: center; margin: 10px 0;">
            <h3>⏰ {st.session_state['current_timer_task']}</h3>
            <h2>{time_str}</h2>
            <div style="background: rgba(255,255,255,0.3); height: 10px; border-radius: 5px;">
                <div style="background: white; width: {progress*100}%; height: 100%; border-radius: 5px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if remaining <= 0:
            st.error("⏰ Time's up! Mark complete or extend")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Mark Complete"):
                    st.session_state["timer_running"] = False
                    st.success("Task marked complete!")
                    st.rerun()
            with col2:
                if st.button("➕ Add 15 mins"):
                    st.session_state["timer_end_time"] += 15 * 60
                    st.info("Extended by 15 minutes")
        
        time.sleep(1)
        st.rerun()


# Feature 9: Streak Tracker
def update_streak_tracker(plan):
    day_plan = plan.get("day_plan", {})
    completed_tasks = st.session_state.get("completed_tasks", {})
    
    # Check if current day is complete
    current_day = st.session_state["current_day"]
    day_key = f"day_{current_day}"
    day_tasks = day_plan.get(day_key, {}).get("tasks", [])
    completed_indices = completed_tasks.get(day_key, [])
    
    is_day_complete = len(completed_indices) == len(day_tasks) and len(day_tasks) > 0
    
    if is_day_complete:
        current_streak = st.session_state.get("streak", 0) + 1
        st.session_state["streak"] = current_streak
        st.session_state["best_streak"] = max(st.session_state.get("best_streak", 0), current_streak)
    else:
        # Reset streak if day is not complete and it's a new day
        if st.session_state.get("last_checked_day", 0) != current_day:
            st.session_state["streak"] = 0
    
    st.session_state["last_checked_day"] = current_day


# Feature 12: Stuck? Get Help
def get_stuck_help(task, plan):
    client = LLMClient("groq")
    if not client.is_configured():
        st.error("LLM not configured for help")
        return
    
    prompt = f"""
Task: {task.get('summary', '')}
Project: {plan.get('project_idea', '')}
Skill level: {plan.get('skill_level', 'Intermediate')}

The user is stuck. Help them by:
1. Explaining the task in simpler terms
2. Listing 3 common mistakes beginners make
3. Giving exact steps to get unstuck
4. Providing a minimal code example if applicable

Be encouraging and specific.
"""
    
    try:
        response = client.call_llm(prompt=prompt, system_prompt="Help stuck developers with clear, encouraging guidance.")
        
        st.subheader("🆘 Help for: " + task.get('summary', ''))
        st.markdown(response)
        
        # Break into smaller steps option
        if st.button("🔨 Break into Smaller Steps", key=f"break_{hash(task.get('summary', ''))}"):
            break_into_steps(task, plan)
            
    except Exception as e:
        st.error(f"Failed to get help: {e}")


def break_into_steps(task, plan):
    client = LLMClient("groq")
    if not client.is_configured():
        return
    
    prompt = f"""
Break this task into 3 micro-steps:
Task: {task.get('summary', '')}
Project: {plan.get('project_idea', '')}

Format as:
1. First micro-step (5-10 mins)
2. Second micro-step (5-10 mins)  
3. Third micro-step (5-10 mins)
"""
    
    try:
        response = client.call_llm(prompt=prompt, system_prompt="Break tasks into small, achievable steps.")
        st.markdown("**Micro-steps:**")
        st.markdown(response)
    except Exception as e:
        st.error(f"Failed to break into steps: {e}")


# Feature 15: Confetti Animation
def check_project_completion(plan):
    day_plan = plan.get("day_plan", {})
    completed_tasks = st.session_state.get("completed_tasks", {})
    
    total_tasks = 0
    completed_count = 0
    
    for day_key, day_data in day_plan.items():
        tasks = day_data.get("tasks", [])
        total_tasks += len(tasks)
        completed_count += len(completed_tasks.get(day_key, []))
    
    if total_tasks > 0 and completed_count == total_tasks:
        # Project complete!
        show_confetti()
        
        st.balloons()
        st.success("🎉 Project Complete! You did it!")
        
        # Stats
        total_days = len(day_plan)
        streak = st.session_state.get("streak", 0)
        difficulty = plan.get("difficulty_score", 5)
        
        st.markdown(f"""
        ### 📊 Completion Stats
        - **Total Days:** {total_days}
        - **Total Tasks Completed:** {completed_count}
        - **Final Streak:** {streak} days
        - **Difficulty Conquered:** {difficulty}/10
        """)
        
        if st.button("🎊 Celebrate Again!", use_container_width=True):
            show_confetti()


def show_confetti():
    confetti_html = """
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
    <script>
    confetti({
        particleCount: 200,
        spread: 70,
        origin: { y: 0.6 }
    });
    </script>
    """
    st.components.v1.html(confetti_html, height=0)


if __name__ == "__main__":
    main()
