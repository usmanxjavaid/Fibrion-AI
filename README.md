# Fibrion

**Production intelligence for textile manufacturing.**

Fibrion ingests raw production data from textile manufacturing processes,
runs it through a chain of specialized AI agents that clean it, compute
KPIs, detect anomalies, and generate a professional analysis, then
delivers a report to the channel you choose — Telegram, email, or
(later) WhatsApp — with a web dashboard for deeper exploration.

The system is built module by module: **weaving is the first process
module**, validated against real production data, with spinning,
dyeing/finishing, and garment manufacturing designed to follow as
additional pluggable modules on the same architecture.

## Status

🚧 **Phase 1 — core pipeline validation.** No database, no dashboard, no
auth yet. The current goal is proving the agent pipeline produces
correct, trustworthy output against real weaving production data before
anything else gets built.

## Architecture

A LangGraph state graph orchestrates eight agents, each with one job —
ingestion, validation, KPI calculation, analysis, visualization, report
generation, verification, and notification. Any step with exactly one
correct numeric answer stays in plain code; LLMs are used only for
genuinely linguistic or judgment-based tasks.

Every process type (weaving, and later spinning/dyeing/garment) is a
pluggable module under `backend/core/schema_registry/`, implementing a
shared interface — the eight agents are process-agnostic and never
hardcode weaving-specific logic directly.

## Tech stack

- **Backend:** FastAPI, LangGraph orchestration
- **LLM API:** OpenRouter (Claude Haiku 4.5 for narrow tasks, Claude Sonnet 5 for reasoning tasks)
- **Observability:** LangSmith tracing
- **Data processing:** pandas, NumPy
- **Reporting:** ReportLab (PDF), matplotlib (charts)
- **Delivery:** Telegram Bot API, Gmail SMTP; WhatsApp planned
- **Frontend (phase 3):** Next.js, TypeScript, Recharts
- **Persistence (phase 2):** PostgreSQL, SQLAlchemy

## Local development

```bash
cd backend
cp .env.example .env      # fill in your real keys
pip install -r requirements.txt
```

## License

MIT — see [LICENSE](LICENSE).