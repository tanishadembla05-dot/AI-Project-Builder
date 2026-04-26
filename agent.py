from __future__ import annotations

import json
import re
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from dataset import load_project_templates, load_risk_database, load_tech_stacks
from llm_client import LLMClient
from utils import (
    clean_text,
    ensure_daywise_non_repetitive,
    keyword_similarity,
    make_report_summary,
    safe_json_loads,
    timeline_phase,
    unique_preserve,
)


def _task_obj(summary: str, what: str, how: str, output: str, time: str) -> Dict[str, str]:
    return {
        "summary": summary,
        "what": what,
        "how": how,
        "output": output,
        "time": time,
    }


def _normalize_tasks(tasks: Any) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    if not isinstance(tasks, list):
        return normalized
    for item in tasks:
        if isinstance(item, dict):
            summary = str(item.get("summary", "")).strip() or "Task"
            normalized.append(
                _task_obj(
                    summary=summary,
                    what=str(item.get("what", summary)),
                    how=str(item.get("how", "")),
                    output=str(item.get("output", "")),
                    time=str(item.get("time", "45 mins")),
                )
            )
        else:
            text = str(item).strip()
            normalized.append(
                _task_obj(
                    summary=text or "Task",
                    what=text or "Complete this task.",
                    how="Use your standard development workflow and verify with a quick check.",
                    output="Task completed with visible proof in app/terminal.",
                    time="45 mins",
                )
            )
    return normalized


def _parse_minutes(value: str) -> int:
    text = (value or "").strip().lower()
    if not text:
        return 45
    h_match = re.search(r"(\d+)\s*(h|hr|hrs|hour|hours)", text)
    m_match = re.search(r"(\d+)\s*(m|min|mins|minute|minutes)", text)
    if h_match and m_match:
        return int(h_match.group(1)) * 60 + int(m_match.group(1))
    if h_match:
        return int(h_match.group(1)) * 60
    if m_match:
        return int(m_match.group(1))
    num = re.search(r"\d+", text)
    return int(num.group(0)) if num else 45


def _minutes_to_label(minutes: int) -> str:
    minutes = max(20, minutes)
    if minutes % 60 == 0:
        return f"{minutes // 60} hour" if minutes == 60 else f"{minutes // 60} hours"
    if minutes > 60:
        h = minutes // 60
        m = minutes % 60
        return f"{h}h {m} mins"
    return f"{minutes} mins"


def _task_total_minutes(tasks: List[Dict[str, str]]) -> int:
    return sum(_parse_minutes(task.get("time", "45 mins")) for task in tasks)


def _fill_day_to_time_budget(day_payload: Dict[str, Any], state: "AgentState", day_num: int) -> Dict[str, Any]:
    tasks = _normalize_tasks(day_payload.get("tasks", []))
    target = int(state.get("daily_hours", 4)) * 60
    lower_bound = max(60, target - 50)
    current = _task_total_minutes(tasks)
    if current >= lower_bound:
        day_payload["tasks"] = tasks
        return day_payload

    project_idea = state.get("project_idea", "project")
    phase = day_payload.get("phase", "Development")
    tech_stack = state.get("matched_template", {}).get("tech_stack", [])
    primary_tool = tech_stack[0] if tech_stack else "Python"
    secondary_tool = tech_stack[1] if len(tech_stack) > 1 else "Streamlit"
    patterns = [
        {
            "mins": 75,
            "summary": f"Implement core {phase.lower()} code path in `src/day_{day_num}_feature.py`",
            "what": f"Write or extend the main logic for '{project_idea}' in a dedicated module for Day {day_num}.",
            "how": f"Create `src/day_{day_num}_feature.py`, add `build_day_{day_num}_feature()` using {primary_tool}, and run local test command.",
            "output": f"Module compiles and produces expected output for one realistic {project_idea} scenario.",
        },
        {
            "mins": 50,
            "summary": f"Test critical flow with CLI/API checks for Day {day_num}",
            "what": "Verify the key feature path with positive and negative test cases.",
            "how": f"Run `python -m pytest` (or equivalent) and capture response/trace details for failed and passing cases.",
            "output": "Test evidence file with pass/fail status and corrected issues.",
        },
        {
            "mins": 25,
            "summary": f"Document execution notes in `notes/day_{day_num}_execution.md`",
            "what": "Record exact implementation steps, commands, and unresolved edge cases.",
            "how": f"Write notes including commands, function names, and outcomes using {secondary_tool} workflow context.",
            "output": "Review-ready day log that another teammate can reproduce end-to-end.",
        },
    ]
    idx = 0
    while current < lower_bound:
        template = patterns[idx % len(patterns)]
        remaining = target - current
        mins = min(template["mins"], remaining if remaining > 20 else template["mins"])
        tasks.append(
            _task_obj(
                summary=template["summary"],
                what=template["what"],
                how=template["how"],
                output=template["output"],
                time=_minutes_to_label(mins),
            )
        )
        current = _task_total_minutes(tasks)
        idx += 1

    day_payload["tasks"] = tasks
    return day_payload


class AgentState(TypedDict, total=False):
    project_idea: str
    extra_features: str
    days: int
    skill_level: str
    team_size: int
    daily_hours: int
    provider: str
    category: str
    difficulty_score: int
    confidence_percentage: int
    tools: List[str]
    risks: List[Dict[str, str]]
    milestones: List[str]
    deliverables: List[str]
    day_plan: Dict[str, Dict[str, Any]]
    mvp_features: List[str]
    optional_features: List[str]
    viva_summary: str
    report_summary: str
    viva_questions: List[Dict[str, str]]
    matched_template: Dict[str, Any]
    errors: List[str]


def input_validator(state: AgentState) -> AgentState:
    try:
        state["project_idea"] = clean_text(state.get("project_idea", "Untitled Project"))
        state["extra_features"] = clean_text(state.get("extra_features", ""))
        state["days"] = max(1, min(90, int(state.get("days", 7))))
        state["team_size"] = max(1, min(10, int(state.get("team_size", 1))))
        state["daily_hours"] = max(1, min(12, int(state.get("daily_hours", 4))))
        skill = str(state.get("skill_level", "Beginner")).title()
        state["skill_level"] = skill if skill in {"Beginner", "Intermediate", "Advanced"} else "Beginner"
        state["errors"] = state.get("errors", [])
    except Exception as exc:
        state["errors"] = state.get("errors", []) + [f"input_validator: {exc}"]
    return state


def dataset_matcher(state: AgentState) -> AgentState:
    try:
        templates = load_project_templates()
        project_idea = state.get("project_idea", "")
        best_match = None
        best_score = -1.0
        for tpl in templates:
            haystack = " ".join(
                [
                    tpl.get("name", ""),
                    tpl.get("category", ""),
                    " ".join(tpl.get("tech_stack", [])),
                    " ".join(tpl.get("similar_projects", [])),
                    " ".join(tpl.get("mvp_features", [])),
                ]
            )
            score = keyword_similarity(project_idea, haystack)
            if score > best_score:
                best_score = score
                best_match = tpl
        if not best_match and templates:
            best_match = templates[0]
        state["matched_template"] = best_match or {}
    except Exception as exc:
        state["errors"] = state.get("errors", []) + [f"dataset_matcher: {exc}"]
    return state


def project_analyzer(state: AgentState) -> AgentState:
    try:
        template = state.get("matched_template", {})
        client = LLMClient(state.get("provider"))
        fallback = {
            "category": template.get("category", "General"),
            "difficulty_score": int(template.get("difficulty", 5)),
            "confidence_percentage": max(50, 95 - abs(state["days"] - int(template.get("max_days", 30)))),
        }
        if not client.is_configured():
            state.update(fallback)
            return state

        prompt = f"""
Analyze this project idea and return strict JSON:
{{
  "category": "...",
  "difficulty_score": 1-10,
  "confidence_percentage": 1-100
}}

Project idea: {state["project_idea"]}
Extra requested features: {state.get("extra_features", "None")}
Timeline days: {state["days"]}
Skill level: {state["skill_level"]}
Team size: {state["team_size"]}
Daily hours: {state["daily_hours"]}
Matched template category: {template.get("category", "")}
"""
        out = client.call_llm(prompt=prompt, system_prompt="You are a precise project analyst. Return only JSON.")
        parsed = safe_json_loads(out, fallback)
        state["category"] = str(parsed.get("category", fallback["category"]))
        state["difficulty_score"] = max(1, min(10, int(parsed.get("difficulty_score", fallback["difficulty_score"]))))
        state["confidence_percentage"] = max(
            1, min(100, int(parsed.get("confidence_percentage", fallback["confidence_percentage"])))
        )
    except Exception as exc:
        state["errors"] = state.get("errors", []) + [f"project_analyzer: {exc}"]
    return state


def _dataset_day_plan(state: AgentState) -> Dict[str, Dict[str, Any]]:
    plan: Dict[str, Dict[str, Any]] = {}
    total_days = state["days"]
    idea = state["project_idea"]
    for d in range(1, total_days + 1):
        phase = timeline_phase(total_days, d)
        plan[f"day_{d}"] = {
            "phase": phase,
            "tasks": [
                _task_obj(
                    summary=f"Define scope and acceptance checks for Day {d}",
                    what=f"Map the day goals for '{idea}' into 4-5 concrete acceptance checks.",
                    how=f"Create notes/day_{d}_scope.md and list measurable checks aligned with the current phase.",
                    output="A clear day scope document with acceptance checklist ready for execution.",
                    time="30 mins",
                ),
                _task_obj(
                    summary=f"Implement the primary {phase.lower()} component",
                    what=f"Build the highest-priority feature for Day {d} tied to the project milestone.",
                    how="Code the component using the selected stack, run a targeted smoke test, and capture logs/screenshots.",
                    output="Working feature implementation with test evidence and reproducible run steps.",
                    time="90 mins",
                ),
                _task_obj(
                    summary=f"Document progress and blockers for Day {d}",
                    what="Summarize completed work, pending items, and blockers found during execution.",
                    how=f"Update notes/day_{d}_report.md with bullets, error traces, and next-day handoff notes.",
                    output="A review-ready daily report that can be used in demo/viva discussion.",
                    time="25 mins",
                ),
            ],
            "duration_estimate": f"{state['daily_hours']} hours",
            "key_output": f"{phase} deliverable checkpoint {d}",
        }
    return ensure_daywise_non_repetitive(plan)


def plan_generator(state: AgentState) -> AgentState:
    try:
        template = state.get("matched_template", {})
        dataset_plan = _dataset_day_plan(state)
        client = LLMClient(state.get("provider"))

        if not client.is_configured():
            for day_key, day_payload in dataset_plan.items():
                day_num_match = re.search(r"\d+", day_key)
                day_num = int(day_num_match.group(0)) if day_num_match else 1
                dataset_plan[day_key] = _fill_day_to_time_budget(day_payload, state, day_num)
            state["day_plan"] = dataset_plan
            return state

        prompt = f"""
Generate a strict JSON object for day-wise project plan.
- Days: {state["days"]}
- Skill level: {state["skill_level"]}
- Team size: {state["team_size"]}
- Daily hours: {state["daily_hours"]}
- Project idea: {state["project_idea"]}
- Extra requested features: {state.get("extra_features", "None")}
- Template milestones: {template.get("milestones", [])}
- Phases must scale:
  * 1-3 days: simple execution
  * 4-14 days: Planning -> Dev -> Testing -> Polish -> Demo
  * 15+ days: full SDLC style with deeper breakdown
- Make every day non-repetitive.
Format:
{{
  "day_1": {{"phase":"...","tasks":["..."],"duration_estimate":"...","key_output":"..."}},
  "day_2": ...
}}
Return JSON only.
"""
        task_system_prompt = """
You are a senior software engineer creating a day-wise project plan.
Each task must be:
- Specific and actionable (not vague)
- Include WHAT to do, HOW to do it, and WHAT the output should be
- Relevant to the actual project idea provided
- Must incorporate requested extra features/constraints when provided
- Realistic for the given skill level
- Different from tasks on other days (no repetition)
- Explicitly mention concrete files, functions/classes, and runnable commands
- Must use the actual project idea and selected tech stack names in every task
- Never use placeholders like "the project", "module", "component", "day task"

Task format:
"[Action verb] [specific component] using [specific tool/method]
 -> Output: [concrete deliverable]"

Each task object must be:
{{
  "summary": "Set up project folder structure",
  "what": "Create the base folder with all required files",
  "how": "Open terminal and run: mkdir data && touch app.py agent.py requirements.txt",
  "output": "All files visible in VS Code explorer panel",
  "time": "20 mins"
}}

Examples:
- "Create a Streamlit sidebar with 3 input fields (text, slider, dropdown)
   using st.sidebar -> Output: working input form visible in browser"
- "Write the LangGraph StateGraph with 3 nodes (analyzer, planner, formatter)
   and test the flow with a hardcoded input -> Output: terminal prints
   final state dict"
- "Design the JSON schema for project_templates.json with 5 sample entries
   covering AI, Web, and Mobile categories -> Output: valid JSON file saved"

Never write tasks like:
- "Work on project module"
- "Development task for day X"
- "Team sync and update"
- "Validate outputs"
These are meaningless. Every task must be concrete.
Return only valid JSON.
"""
        out = client.call_llm(prompt=prompt, system_prompt=task_system_prompt)
        parsed = safe_json_loads(out, dataset_plan)
        if not isinstance(parsed, dict) or not parsed:
            parsed = dataset_plan
        for day_key, day_payload in parsed.items():
            if isinstance(day_payload, dict):
                day_payload["tasks"] = _normalize_tasks(day_payload.get("tasks", []))
                day_num_match = re.search(r"\d+", day_key)
                day_num = int(day_num_match.group(0)) if day_num_match else 1
                day_payload = _fill_day_to_time_budget(day_payload, state, day_num)
                parsed[day_key] = day_payload
        state["day_plan"] = ensure_daywise_non_repetitive(parsed)
    except Exception as exc:
        fallback = _dataset_day_plan(state)
        for day_key, day_payload in fallback.items():
            day_num_match = re.search(r"\d+", day_key)
            day_num = int(day_num_match.group(0)) if day_num_match else 1
            fallback[day_key] = _fill_day_to_time_budget(day_payload, state, day_num)
        state["day_plan"] = fallback
        state["errors"] = state.get("errors", []) + [f"plan_generator: {exc}"]
    return state


def feature_classifier(state: AgentState) -> AgentState:
    try:
        template = state.get("matched_template", {})
        default_mvp = template.get("mvp_features", ["Core functionality"])
        default_optional = template.get("optional_features", ["Enhancements"])
        client = LLMClient(state.get("provider"))

        if not client.is_configured():
            state["mvp_features"] = default_mvp
            state["optional_features"] = default_optional
            return state

        prompt = f"""
Given this project idea, classify features into MVP and Optional.
Project: {state["project_idea"]}
Extra requested features: {state.get("extra_features", "None")}
Existing hints:
- MVP: {default_mvp}
- Optional: {default_optional}
Return strict JSON:
{{
  "mvp_features": ["..."],
  "optional_features": ["..."]
}}
"""
        out = client.call_llm(prompt=prompt, system_prompt="You are a product strategist. Return JSON only.")
        parsed = safe_json_loads(out, {})
        mvp = parsed.get("mvp_features", default_mvp)
        optional = parsed.get("optional_features", default_optional)
        state["mvp_features"] = unique_preserve(mvp)[:10]
        state["optional_features"] = unique_preserve(optional)[:12]
    except Exception as exc:
        state["errors"] = state.get("errors", []) + [f"feature_classifier: {exc}"]
    return state


def risk_assessor(state: AgentState) -> AgentState:
    try:
        template = state.get("matched_template", {})
        category = state.get("category") or template.get("category", "AI/GenAI")
        risk_db = load_risk_database()
        category_risks = risk_db.get(category, [])
        base_risks = [{"risk": r, "severity": "Medium"} for r in template.get("risks", [])]

        client = LLMClient(state.get("provider"))
        llm_risks: List[Dict[str, str]] = []
        if client.is_configured():
            prompt = f"""
Generate risks specifically for: {state.get("project_idea", "")}
Tech stack being used: {state.get("tools", [])}
Project category: {category}
Extra features requested: {state.get("extra_features", "None")}

Generate exactly:
- 3 High severity risks
- 4 Medium severity risks  
- 2 Low severity risks

Each risk must:
- Be specific to THIS project and its tech stack
- Name actual components that could fail
- Not be copy-pasted generic AI risks unless truly relevant

Return JSON:
{{
  "high": ["risk1", "risk2", "risk3"],
  "medium": ["risk1", "risk2", "risk3", "risk4"],
  "low": ["risk1", "risk2"]
}}
"""
            out = client.call_llm(prompt=prompt, system_prompt="Return JSON only.")
            parsed = safe_json_loads(out, {})
            if isinstance(parsed, dict):
                # Convert to the expected format
                for severity, risks_list in parsed.items():
                    if isinstance(risks_list, list):
                        for risk in risks_list:
                            llm_risks.append({"risk": str(risk), "severity": severity.capitalize()})

        merged = unique_preserve(category_risks + base_risks + llm_risks)[:12]
        state["risks"] = merged
    except Exception as exc:
        state["errors"] = state.get("errors", []) + [f"risk_assessor: {exc}"]
    return state


def output_formatter(state: AgentState) -> AgentState:
    try:
        template = state.get("matched_template", {})
        category = state.get("category", template.get("category", "General"))
        tech_stacks = load_tech_stacks()
        skill_tools = tech_stacks.get(category, {}).get(state.get("skill_level", "Beginner"), [])
        tools = unique_preserve(template.get("tech_stack", []) + skill_tools)

        state["tools"] = tools
        
        # Generate project-specific milestones
        client = LLMClient(state.get("provider"))
        if client.is_configured():
            milestone_prompt = f"""
Generate 4-5 milestones specifically for: {state.get("project_idea", "")}
Timeline: {state.get("days", 5)} days
Extra features: {state.get("extra_features", "None")}
Tech stack: {tools}

Each milestone must name an actual feature of THIS project.

BAD: "Basic chat working" for a travel app
GOOD: "Flight search and display working"
GOOD: "Budget calculator giving accurate estimates"
GOOD: "Hotel recommendations loading from API"

Return as JSON list of strings.
"""
            milestone_out = client.call_llm(prompt=milestone_prompt, system_prompt="Return JSON array only.")
            milestone_parsed = safe_json_loads(milestone_out, [])
            if isinstance(milestone_parsed, list) and milestone_parsed:
                state["milestones"] = [str(m) for m in milestone_parsed[:5]]
            else:
                state["milestones"] = template.get("milestones", ["Requirement freeze", "MVP complete", "Demo ready"])
        else:
            state["milestones"] = template.get("milestones", ["Requirement freeze", "MVP complete", "Demo ready"])
        
        state["deliverables"] = template.get("deliverables", ["Working demo", "GitHub repo", "README", "PPT"])
        viva_questions = template.get("viva_questions", [])
        state["viva_summary"] = (
            f"Focus on explaining architecture decisions, trade-offs, and risk handling for {state['project_idea']}."
        )
        state["viva_questions"] = [
            {"question": q, "short_answer": "Prepare a concise 3-4 line technical explanation with one example."}
            for q in viva_questions[:5]
        ]
        state["report_summary"] = make_report_summary(
            state["project_idea"], category, state.get("difficulty_score", 5), state.get("confidence_percentage", 70)
        )
    except Exception as exc:
        state["errors"] = state.get("errors", []) + [f"output_formatter: {exc}"]
    return state


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("input_validator", input_validator)
    graph.add_node("dataset_matcher", dataset_matcher)
    graph.add_node("project_analyzer", project_analyzer)
    graph.add_node("plan_generator", plan_generator)
    graph.add_node("feature_classifier", feature_classifier)
    graph.add_node("risk_assessor", risk_assessor)
    graph.add_node("output_formatter", output_formatter)

    graph.set_entry_point("input_validator")
    graph.add_edge("input_validator", "dataset_matcher")
    graph.add_edge("dataset_matcher", "project_analyzer")
    graph.add_edge("project_analyzer", "plan_generator")
    graph.add_edge("plan_generator", "feature_classifier")
    graph.add_edge("feature_classifier", "risk_assessor")
    graph.add_edge("risk_assessor", "output_formatter")
    graph.add_edge("output_formatter", END)
    return graph.compile()


def run_agent(
    project_idea: str,
    extra_features: str,
    days: int,
    skill_level: str,
    team_size: int,
    daily_hours: int,
    provider: str = "groq",
) -> AgentState:
    app = build_graph()
    initial_state: AgentState = {
        "project_idea": project_idea,
        "extra_features": extra_features,
        "days": days,
        "skill_level": skill_level,
        "team_size": team_size,
        "daily_hours": daily_hours,
        "provider": provider,
    }
    result = app.invoke(initial_state)
    # Ensure serializable state
    json.dumps(result)
    return result
