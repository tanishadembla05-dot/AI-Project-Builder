# AI Project Builder Agent

Professional planning-based AI agent that converts an idea into a complete execution roadmap with day-wise tasks, MVP scoping, risk analysis, milestones, viva prep, and exportable reports.

## Tech Stack

- Python 3.10+
- Streamlit UI
- LangGraph workflow orchestration
- Groq (`llama3-8b-8192`) or Gemini (`gemini-1.5-flash`) for reasoning
- `fpdf2` for PDF export
- JSON datasets for templates, tools, and risks

## Features

- Professional day-wise planning that scales by timeline:
  - 1-3 days: compact execution
  - 4-14 days: phased delivery
  - 15+ days: full SDLC progression
- 50 project templates across 10 domains
- 110+ categorized risks with severity
- MVP vs Optional feature classification
- Completion confidence and difficulty scoring
- Dark premium Streamlit dashboard with tabbed insights
- Progress tracker with completed-day persistence
- Save/load plans locally from `saved_plans/`
- Download plan as JSON and PDF
- Dataset-only fallback mode (works even without API keys)

## Project Structure

```text
ai_project_builder/
├── app.py
├── agent.py
├── llm_client.py
├── dataset.py
├── utils.py
├── pdf_exporter.py
├── data/
│   ├── project_templates.json
│   ├── tech_stacks.json
│   └── risk_database.json
├── saved_plans/
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. Clone or copy the project folder.
2. Create and activate a Python 3.10+ virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment:

```bash
cp .env.example .env
```

Then add your key(s) in `.env`:
- `GROQ_API_KEY=...` and/or
- `GEMINI_API_KEY=...`
- `LLM_PROVIDER=groq` or `gemini`

5. Run:

```bash
streamlit run app.py
```

## Free API Keys

### Groq (Free Tier)

1. Go to [console.groq.com](https://console.groq.com)
2. Sign in and create an API key
3. Put it in `.env` as `GROQ_API_KEY`

### Gemini (Free Tier)

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Create an API key
3. Put it in `.env` as `GEMINI_API_KEY`

## Notes on Fallback Mode

If API keys are missing or a provider call fails:
- The app still generates a complete plan from local datasets.
- All graph nodes continue with graceful defaults.
- You can still export to JSON/PDF and save plans.

## Screenshots

- `docs/screenshots/overview.png` (placeholder)
- `docs/screenshots/day-plan.png` (placeholder)
- `docs/screenshots/risks.png` (placeholder)

## Future Scope

- Add HuggingFace dataset ingestion for dynamic template updates
- Add team role-based workload distribution
- Add Gantt-style timeline visualization
- Add test-case and architecture diagram generation
- Add multi-language planning output

