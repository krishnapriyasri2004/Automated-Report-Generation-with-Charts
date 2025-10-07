# AI Report Generator — Technical Documentation and Setup

This repository implements an AI-powered report generation system that ingests internal documents, web sources, and email complaints; synthesizes grounded insights with citations; generates charts; and exports polished reports (PDF/DOCX/HTML). It is designed for reliability, observability, and extensibility, using a tool-using LLM agent and a Retrieval-Augmented Generation (RAG) pipeline.

- Tech foundations: Next.js App Router (Next.js), Vercel AI SDK, Postgres (Neon/Supabase), Slack notifications, optional Redis queue, server-side chart rendering.
- Key goals: accuracy via grounding/citations, graceful failure handling across integrations, and production-grade code standards.

---

## 1) AI Architecture

### 1.1 Components
- Ingestion
  - Document ingestion (PDF/DOCX/CSV/XLSX) → text + tables extraction, chunking, embeddings
  - Web retrieval (search + fetch, HTML → clean text with headings)
  - Email ingestion (Gmail/MS Graph/IMAP) for complaints/requests + attachments
- Data Layer
  - Vector store (pgvector or external) for RAG
  - Postgres (Neon/Supabase) metadata for runs, documents, chunks, charts, artifacts, logs
  - Object/asset storage for charts and exports
- Agent & Orchestration
  - LLM agent (Vercel AI SDK) with tools: search, fetch, db, charts, notify
  - Queue (optional, e.g., Upstash Redis) for long-running jobs and retries
- Reporting
  - Section blueprints (Executive Summary, Insights, Risks, Recs)
  - Visualization generator (static images or interactive)
  - Exporters: PDF, DOCX, HTML Dashboard
- Notifications
  - Slack/Teams/Discord updates for run lifecycle
- Observability
  - Structured logging, metrics, traces, audit of model/tool calls

### 1.2 End-to-end Data Flow
1) Trigger (user request or schedule) creates a run
2) Ingestion pulls data from internal docs, email, and the web
3) Indexing embeds chunks and stores metadata
4) Planning: agent drafts section plan
5) Retrieval: top-K context via vector search + filters
6) Synthesis: LLM generates grounded sections with citations
7) Visualization: chart service renders images (or interactive)
8) Assembly: exporter builds PDF/DOCX/HTML with references
9) Delivery: artifacts stored and shared; Slack notifies channel
10) Feedback: edits/ratings logged for continuous improvement

### 1.3 Architecture Diagram (Mermaid)
\`\`\`mermaid
flowchart LR
  UI[Client/UI]  API[API /runs, /artifacts]
  API  Q[Queue (optional)]
  Q  ING[Ingestion Workers]
  ING  VDB[(Vector DB)]
  ING  PG[(Postgres)]
  Q  AG[Agent Orchestrator]
  AG  VDB
  AG  CH[Chart Renderer]
  CH  ASSETS[(Assets Store)]
  AG  EX[Exporter]
  EX  ASSETS
  EX  PG
  API  SLACK((Slack))
  EX  API
\`\`\`

### 1.4 Sequence Diagram
\`\`\`mermaid
sequenceDiagram
  participant U as User
  participant API as API Server
  participant Q as Queue
  participant ING as Ingestion
  participant AG as Agent
  participant DB as Postgres/Vector
  participant CH as Charts
  participant EX as Exporter
  participant SL as Slack

  U->>API: POST /runs (topic, audience, tone)
  API->>Q: enqueue(runId)
  Q->>ING: start ingestion
  ING->>DB: store docs + embeddings
  ING->>Q: ingestion_done(runId)
  Q->>AG: start generation
  AG->>DB: retrieve context (RAG)
  AG->>CH: render charts (spec + data)
  CH>AG: chart image URL
  AG->>EX: assemble sections + assets
  EX->>DB: persist artifact metadata
  EX->>API: artifact URL
  API->>SL: notify success/failure
\`\`\`



## 2) Integration & Implementation

### 2.1 Email Service (Reading + Processing Complaints)
- Providers: Gmail API (OAuth2), Microsoft Graph (Outlook 365), IMAP
- Strategy:
  - Poll or webhook from a dedicated folder/label
  - Extract HTML/text; capture message metadata + attachments
  - Persist to Postgres tables: emails, email_attachments; link to runs
- Failure handling:
  - Exponential backoff, circuit breaker, DLQ (if using Redis)
  - If email provider unavailable, proceed without email source and mark run as “degraded”; notify Slack

### 2.2 Web Search & Fetch
- Use a provider (e.g., Bing/Serper/Tavily); deduplicate results
- Extract with readability + boilerplate removal; store title, URL, publish/access dates
- Track citations for every used fact

### 2.3 Charts & Visualizations
- Server-side static images for portability (e.g., Chart.js + node-canvas, Vega-Lite SSR)
- Optional interactive dashboard (Recharts) for web UI
- Heuristics to select chart types: time series → line; categorical → bar; proportion → pie; correlation → scatter

### 2.4 Export
- PDF/DOCX exporters with templates and references appendix
- HTML dashboard variant with interactive charts
- Store artifact URI + metadata in DB; expose via /artifacts

---

## 3) Code Standards

### 3.1 Conventions
- Naming: lowerCamelCase (vars/functions), PascalCase (types/classes), SNAKE_CASE (env vars)
- Errors & Logging:
  - Typed errors, context-rich messages; never swallow errors
  - Structured logs (JSON), redact secrets/PII; include runId, step, tool, duration, sizes
- Config & Secrets:
  - Never hardcode secrets; use environment variables
  - In Next.js: env is available on the server only; client-side requires NEXT_PUBLIC_ prefix
- Comments & Docs:
  - Docstrings for public functions; rationale for complex logic
- Security:
  - Principle of least privilege; validate/parse inputs (e.g., zod)
  - RLS if using Supabase; encrypt artifacts at rest where applicable
- Testing:
  - Unit tests (parsers, chunkers, retriever ranking, spec validators)
  - Integration tests (email, Slack, DB migrations)
  - E2E (happy path run + citations assertions)



## 6) Configuration

Set environment variables in your project settings (never commit secrets):
- DATABASE_URL=postgres://...
- OPENAI_API_KEY=...
- ANTHROPIC_API_KEY=... (if used)



