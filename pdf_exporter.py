from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fpdf import FPDF


class ReportPDF(FPDF):
    def __init__(self, project_name: str, report_date: str):
        super().__init__()
        self.project_name = project_name
        self.report_date = report_date

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 6, f"{self.project_name} | {self.report_date}", ln=True, align="R")
        self.ln(1)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


class PDFExporter:
    def __init__(self):
        self.report_date = datetime.now().strftime("%Y-%m-%d")
        self.pdf: ReportPDF | None = None

    def _clean_text(self, value: Any) -> str:
        text = str(value) if value is not None else ""
        for ch in "{}[]'":
            text = text.replace(ch, "")
        return text.encode("latin-1", errors="replace").decode("latin-1")

    def _section(self, heading: str):
        self.pdf.set_font("Helvetica", "BU", 14)
        self.pdf.set_text_color(15, 60, 100)
        self.pdf.cell(0, 9, self._clean_text(heading), ln=True)
        self.pdf.set_draw_color(0, 180, 220)
        y = self.pdf.get_y()
        self.pdf.line(self.pdf.l_margin, y, self.pdf.w - self.pdf.r_margin, y)
        self.pdf.ln(3)
        self.pdf.set_font("Helvetica", "", 10)
        self.pdf.set_text_color(20, 20, 20)

    def _line(self, text: str, h: float = 6):
        self.pdf.set_x(self.pdf.l_margin)
        self.pdf.multi_cell(0, h, self._clean_text(text))

    def _normalize_task(self, task: Any) -> Dict[str, str]:
        if isinstance(task, dict):
            return {
                "summary": self._clean_text(task.get("summary", "Task")),
                "what": self._clean_text(task.get("what", "")),
                "how": self._clean_text(task.get("how", "")),
                "output": self._clean_text(task.get("output", "")),
                "time": self._clean_text(task.get("time", "")),
            }
        text = self._clean_text(task)
        return {"summary": text, "what": text, "how": "", "output": "", "time": ""}

    def _mitigation(self, risk_text: str, severity: str) -> str:
        sev = severity.lower()
        if sev == "high":
            return "Immediate mitigation owner, add fallback path, and test weekly."
        if sev == "low":
            return "Monitor in review checklist and verify before demo."
        return "Track in sprint board and review in daily standup."

    def _ascii_flowchart(self, plan: Dict[str, Any]) -> List[str]:
        return [
            "[Project Idea]",
            "    |",
            "[Planning] -> [Development] -> [Testing] -> [Polish & Demo]",
            "    |",
            "[Final Deliverables]",
            "    |",
            "[Milestones Achieved]",
        ]

    def _cover_page(self, plan: Dict[str, Any]):
        self.pdf.add_page()
        self.pdf.set_font("Helvetica", "B", 22)
        self.pdf.set_text_color(10, 30, 70)
        self.pdf.multi_cell(0, 12, self._clean_text(plan.get("project_idea", "Untitled Project")))
        self.pdf.ln(4)
        self.pdf.set_font("Helvetica", "", 11)
        self.pdf.set_text_color(20, 20, 20)
        self._line(f"Generated date: {self.report_date}")
        self._line(f"Team size: {plan.get('team_size', 1)}")
        self._line(f"Skill level: {plan.get('skill_level', 'Beginner')}")
        self._line(f"Timeline: {plan.get('days', 1)} days")
        self._line(f"Difficulty score: {plan.get('difficulty_score', 5)}/10")
        self._line(f"Completion confidence: {plan.get('confidence_percentage', 70)}%")
        self.pdf.ln(5)
        self._section("Project Flow (ASCII)")
        for row in self._ascii_flowchart(plan):
            self._line(row)

    def _summary_page(self, plan: Dict[str, Any]):
        self.pdf.add_page()
        self._section("Executive Summary")
        self._line(plan.get("report_summary", "No summary available."))
        self.pdf.ln(2)
        self._section("Tools & Tech Stack")
        for tool in plan.get("tools", []):
            self._line(f"- {tool}")

    def _day_plan_pages(self, plan: Dict[str, Any]):
        day_plan = plan.get("day_plan", {})
        self.pdf.add_page()
        self._section("Day-wise Plan")
        for day_key, info in day_plan.items():
            day_num = day_key.replace("day_", "")
            self.pdf.set_font("Helvetica", "B", 12)
            self._line(f"DAY {day_num} - {info.get('phase', 'Execution')}")
            self.pdf.set_font("Helvetica", "", 10)
            for idx, raw_task in enumerate(info.get("tasks", []), start=1):
                task = self._normalize_task(raw_task)
                self._line(f"Task {idx}: {task.get('summary', '')}")
                self._line("-" * 72)
                if task.get("what"):
                    self._line(f"What: {task['what']}")
                if task.get("how"):
                    self._line(f"How: {task['how']}")
                if task.get("output"):
                    self._line(f"Output: {task['output']}")
                if task.get("time"):
                    self._line(f"Time: {task['time']}")
                self.pdf.ln(1)
            self._line(f"Day duration estimate: {info.get('duration_estimate', '')}")
            self._line(f"Key output: {info.get('key_output', '')}")
            self.pdf.ln(3)
            if self.pdf.get_y() > 240:
                self.pdf.add_page()

    def _features_page(self, plan: Dict[str, Any]):
        self.pdf.add_page()
        self._section("MVP Features")
        for item in plan.get("mvp_features", []):
            self._line(f"- {item}")
        self.pdf.ln(2)
        self._section("Optional Features")
        for item in plan.get("optional_features", []):
            self._line(f"- {item}")

    def _risks_page(self, plan: Dict[str, Any]):
        self.pdf.add_page()
        self._section("Risk Register")
        risks = plan.get("risks", [])
        col_w = [90, 25, 75]
        headers = ["Risk", "Severity", "Mitigation"]
        self.pdf.set_font("Helvetica", "B", 10)
        for i, h in enumerate(headers):
            self.pdf.cell(col_w[i], 8, h, border=1, align="C")
        self.pdf.ln()
        self.pdf.set_font("Helvetica", "", 9)
        for idx, item in enumerate(risks):
            fill = idx % 2 == 0
            self.pdf.set_fill_color(242, 247, 255) if fill else self.pdf.set_fill_color(255, 255, 255)
            if isinstance(item, dict):
                risk_text = self._clean_text(item.get("risk", ""))
                severity = self._clean_text(item.get("severity", "Medium"))
            else:
                risk_text = self._clean_text(item)
                severity = "Medium"
            mitigation = self._mitigation(risk_text, severity)
            self.pdf.cell(col_w[0], 8, risk_text[:55], border=1, fill=fill)
            self.pdf.cell(col_w[1], 8, severity, border=1, align="C", fill=fill)
            self.pdf.cell(col_w[2], 8, mitigation[:48], border=1, fill=fill)
            self.pdf.ln()

    def _milestones_page(self, plan: Dict[str, Any]):
        self.pdf.add_page()
        self._section("Milestones Timeline")
        for idx, milestone in enumerate(plan.get("milestones", []), start=1):
            self._line(f"{idx}. {milestone}")

    def _viva_page(self, plan: Dict[str, Any]):
        self.pdf.add_page()
        self._section("Viva Preparation")
        if plan.get("viva_summary"):
            self._line(plan["viva_summary"])
            self.pdf.ln(2)
        for item in plan.get("viva_questions", []):
            if isinstance(item, dict):
                self._line(f"Q: {item.get('question', '')}")
                self._line(f"A: {item.get('short_answer', '')}")
            else:
                self._line(f"Q: {item}")
            self.pdf.ln(1)

    def _last_page(self, plan: Dict[str, Any]):
        self.pdf.add_page()
        self._section("Report Summary")
        self._line(plan.get("report_summary", "No summary available."))

    def build_pdf(self, plan: Dict[str, Any], output_path: Path) -> Path:
        project_name = self._clean_text(plan.get("project_idea", "AI Project Builder Report"))
        self.pdf = ReportPDF(project_name=project_name, report_date=self.report_date)
        self.pdf.set_auto_page_break(auto=True, margin=15)

        self._cover_page(plan)
        self._summary_page(plan)
        self._day_plan_pages(plan)
        self._features_page(plan)
        self._risks_page(plan)
        self._milestones_page(plan)
        self._viva_page(plan)
        self._last_page(plan)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.pdf.output(str(output_path))
        return output_path
