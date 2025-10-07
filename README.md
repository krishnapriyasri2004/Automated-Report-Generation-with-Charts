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

### 1.5 Model/Technique Choices
- Generation: "openai/gpt-5-mini" or "anthropic/claude-sonnet-4.5" via Vercel AI Gateway
  - Rationale: strong reasoning, long context, reliable tool use
- Embeddings: OpenAI text-embedding-3-large or similar via Gateway
  - Rationale: robust semantic retrieval
- Primary technique: RAG (grounding with citations)
  - Rationale: fresh data without retraining; transparent evidence
- Optional: fine-tuning later for style/format consistency using approved outputs

---

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

Schema sketch:
\`\`\`sql
create table if not exists emails (
  id uuid primary key default gen_random_uuid(),
  provider text not null,
  message_id text not null,
  subject text,
  sender text,
  recipient text,
  received_at timestamptz,
  body_text text,
  body_html text,
  run_id uuid,
  created_at timestamptz default now()
);

create table if not exists email_attachments (
  id uuid primary key default gen_random_uuid(),
  email_id uuid references emails(id) on delete cascade,
  filename text,
  mime_type text,
  storage_url text,
  created_at timestamptz default now()
);
\`\`\`

### 2.2 Database (Storing/Retrieving Queries/Complaints)
- Postgres (Neon/Supabase). Optionally pgvector for embeddings in-DB.
- Core tables:
  - runs, documents, chunks, web_sources, emails, charts, artifacts, logs
- Indices: btree on run_id + created_at, GIN on metadata JSONB, vector index for embeddings if PG-based

Example (subset):
\`\`\`sql
create table if not exists runs (
  id uuid primary key default gen_random_uuid(),
  topic text not null,
  audience text,
  tone text,
  status text not null default 'queued',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists artifacts (
  id uuid primary key default gen_random_uuid(),
  run_id uuid references runs(id) on delete cascade,
  kind text not null,  -- pdf | docx | html
  uri text not null,
  bytes bigint,
  created_at timestamptz not null default now()
);
\`\`\`

### 2.3 Slack/Teams/Discord
- Slack incoming webhook or Bot API
- Post run events: started, ingestion complete, section generation progress, success/failure, artifact ready
- Failures do not block report generation; retry with backoff and log errors

Example (Slack webhook):
\`\`\`ts
export async function sendSlackMessage(text: string, blocks?: any[]) {
  const url = process.env.SLACK_WEBHOOK_URL
  if (!url) throw new Error("Missing SLACK_WEBHOOK_URL")
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text, blocks }),
  })
  if (!res.ok) throw new Error(`Slack error: ${res.status}`)
}
\`\`\`

### 2.4 Web Search & Fetch
- Use a provider (e.g., Bing/Serper/Tavily); deduplicate results
- Extract with readability + boilerplate removal; store title, URL, publish/access dates
- Track citations for every used fact

### 2.5 Charts & Visualizations
- Server-side static images for portability (e.g., Chart.js + node-canvas, Vega-Lite SSR)
- Optional interactive dashboard (Recharts) for web UI
- Heuristics to select chart types: time series → line; categorical → bar; proportion → pie; correlation → scatter

### 2.6 Export
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

### 3.2 Suggested Project Structure (Next.js App Router)
\`\`\`
app/
  api/
    runs/route.ts               # POST create run, GET list
    runs/[id]/route.ts          # GET run status
    artifacts/[id]/route.ts     # GET artifact stream/meta
    webhooks/
      email/route.ts            # email provider webhook
      slack/route.ts            # optional slash commands
  (dashboard and viewer pages)

lib/
  ai/
    planner.ts
    section-writer.ts
    tools/
      search.ts
      fetch.ts
      email.ts
      db.ts
      charts.ts
      notify.ts
  ingestion/
    documents.ts
    web.ts
    email.ts
  services/
    embeddings.ts
    vector-store.ts
    exporter/
      pdf.ts
      docx.ts
      html.ts
  db/
    queries.ts
    schema.sql
  utils/
    logging.ts
    errors.ts
    config.ts
    validation.ts
\`\`\`

### 3.3 AI SDK Usage (Examples)
Use the official examples in this repo for correct usage:
- user_read_only_context/integration_examples/ai_sdk/

Simple text generation:
\`\`\`ts
import { generateText } from "ai"

export async function generateSection(params: {
  topic: string
  audience: string
  tone: string
  retrievedContext: string
}) {
  const prompt = `
You are generating one report section.
Audience: ${params.audience}
Tone: ${params.tone}
Topic: ${params.topic}

Use only the provided context and include inline citations like [1], [2].
Context:
${params.retrievedContext}
`
  const { text } = await generateText({
    model: "openai/gpt-5-mini",
    prompt
  })
  return text
}
\`\`\`

---

## 4) API Surface

- POST /api/runs
  - body: { topic, audience, tone, inputs?: { files[], emailFilters?, webQueries? } }
  - returns: { id, status }
- GET /api/runs/[id]
  - returns: { id, status, progress, sections?, errors? }
- GET /api/artifacts/[id]
  - returns: artifact stream or metadata { kind, uri, size }
- POST /api/webhooks/email
  - handles provider webhooks, stores emails + attachments
- POST /api/webhooks/slack
  - optional slash commands for triggering runs

Error envelope:
\`\`\`json
{ "ok": false, "error": { "code": "INTEGRATION_UNAVAILABLE", "message": "Email provider down", "meta": { "provider": "gmail" } } }
\`\`\`

---

## 5) Database Setup (SQL Snippets)

\`\`\`sql
-- runs
create table if not exists runs (
  id uuid primary key default gen_random_uuid(),
  topic text not null,
  audience text,
  tone text,
  status text not null default 'queued',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- artifacts
create table if not exists artifacts (
  id uuid primary key default gen_random_uuid(),
  run_id uuid references runs(id) on delete cascade,
  kind text not null, -- pdf | docx | html
  uri text not null,
  bytes bigint,
  created_at timestamptz not null default now()
);

-- documents & chunks (for RAG)
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  source_type text not null, -- internal | web | email
  title text,
  uri text,
  mime_type text,
  checksum text,
  created_at timestamptz default now()
);

create table if not exists chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid references documents(id) on delete cascade,
  chunk_idx int not null,
  text text not null,
  embedding_vector vector(1536),   -- if using pgvector
  metadata jsonb default '{}'
);
\`\`\`

---

## 6) Configuration

Set environment variables in your project settings (never commit secrets):
- DATABASE_URL=postgres://...
- OPENAI_API_KEY=...
- ANTHROPIC_API_KEY=... (if used)
- SLACK_WEBHOOK_URL=...
- EMAIL_PROVIDER=gmail|graph|imap
- EMAIL_CLIENT_ID=...
- EMAIL_CLIENT_SECRET=...
- EMAIL_REFRESH_TOKEN=... (or IMAP creds)
- VECTOR_DB_URL=... (if external vector DB)

Notes for Next.js:
- Environment variables are only available on the server
- To expose something to the client, prefix with NEXT_PUBLIC_

---

## 7) Deployment

- v0 Preview and Publish:
  - After pushing, configure environment variables in Project Settings → Environment Variables
  - Publish to Vercel with the button in the v0 UI; the project runs as Next.js app (Next.js)
- GitHub:
  - Use the GitHub button in the top right to push your code to a repo
  - Recommended: store SQL scripts under scripts/ and execute via v0 when needed

---

## 8) Testing

- Unit tests: parsers, chunker, retriever ranking, chart spec validation
- Integration tests: email adapter (sandbox inbox), Slack webhook (mock server), DB migrations
- End-to-end: create run → confirm sections, citations present, artifact generated
- Observability: assert structured logs contain runId, step, durations

---

## 9) Security & Compliance

- Data minimization: ingest only requested sources
- PII: detect and redact in exports when necessary
- Access control: least-privilege tokens and OAuth scopes, audit logs
- Transport/storage: TLS for all traffic, encryption at rest for artifacts if supported
- RLS: enable Row Level Security if using Supabase

---

## 10) Troubleshooting

- Email provider down:
  - Expect degraded runs; Slack will notify; system continues with other sources
- Database unavailable:
  - Queue writes, reduce concurrency; alert; retry with backoff
- Chart rendering failed:
  - Retry; fallback to simpler chart or tabular output
- Slack errors:
  - Do not block report completion; log and continue

---

## References

- Vercel AI SDK examples (in this repo):
  - user_read_only_context/integration_examples/ai_sdk/
- Next.js App Router docs: https://nextjs.org/docs/app
- Supabase: https://supabase.com/ (if used)
- Neon: https://neon.tech/ (if used)
