# 🤖 Northstar Support Assistant — LangGraph Multi-Agent System

> A production-ready, two-agent customer support system built with **LangGraph**, **Google Gemini**, **Qdrant RAG**, and **PostgreSQL** — featuring structured triage, tool-calling automation, and Human-in-the-Loop (HITL) escalation via conditional edges.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Graph Flow](#graph-flow)
- [Agents](#agents)
- [Conditional Edge — Human in the Loop](#conditional-edge--human-in-the-loop)
- [Tools](#tools)
- [RAG System](#rag-system)
- [Database Schema](#database-schema)
- [Prompts](#prompts)
- [Token Tracking](#token-tracking)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Example Conversations](#example-conversations)
- [Key Design Decisions](#key-design-decisions)
- [Dependencies](#dependencies)

---

## Overview

**Northstar Support Assistant** is a multi-agent customer support chatbot for a fictional SaaS company called *Northstar Services*. It is built as a **LangGraph StateGraph** with exactly **two AI agents** and a **Human-in-the-Loop (HITL) conditional edge**.

| Feature | Detail |
|---|---|
| **Framework** | LangGraph (StateGraph) |
| **AI Model** | Google Gemini 2.0 Flash Lite |
| **Embeddings** | Gemini Embedding 2 (3072-dim) |
| **Vector DB** | Qdrant Cloud |
| **Relational DB** | PostgreSQL (customers, invoices, refunds, escalations) |
| **Checkpointing** | PostgreSQL-backed LangGraph checkpointer |
| **HITL** | `interrupt()` + `Command(resume=...)` pattern |
| **Prompts** | 100% externalized in `prompt.yml` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NORTHSTAR SUPPORT SYSTEM                     │
│                                                                 │
│   Customer Message                                              │
│         │                                                       │
│         ▼                                                       │
│   ┌──────────────┐                                              │
│   │  TRIAGE NODE │  ← Agent 1: Gemini structured-output call   │
│   │  (Agent 1)   │    No tools. Returns: category, urgency,    │
│   │              │    sentiment, needs_human, summary           │
│   └──────┬───────┘                                              │
│          │                                                      │
│    ┌─────┴──────────────────┐                                   │
│    │  CONDITIONAL EDGE      │  ← route_after_triage()          │
│    │  (Human-in-the-Loop)   │    THE escalation decision point  │
│    └─────┬──────────────────┘                                   │
│          │                                                      │
│   ┌──────┴──────┐        ┌──────────────────────────┐          │
│   │needs_human  │        │ needs_human = False       │          │
│   │ = True      │        │                           │          │
│   ▼             │        ▼                           │          │
│ ┌────────────┐  │  ┌─────────────────┐               │          │
│ │  HUMAN     │  │  │  REPLY AGENT    │ ← Agent 2:    │          │
│ │  ESCALATE  │  │  │  NODE (Agent 2) │   ReAct tool- │          │
│ │  NODE      │  │  │                 │   calling loop │          │
│ │            │  │  │ Tools:          │               │          │
│ │ 1. Logs to │  │  │ • lookupcustomer│               │          │
│ │    Postgres│  │  │ • list_invoices │               │          │
│ │ 2. Calls   │  │  │ • check_refund  │               │          │
│ │  interrupt()│ │  │ • refund_policy │               │          │
│ │ 3. Suspends│  │  │ • company_policy│               │          │
│ │    graph   │  │  │ • create_refund │               │          │
│ └─────┬──────┘  │  └────────┬────────┘               │          │
│       │         │           │                         │          │
│  Human operator │           │                         │          │
│  types reply    │           │                         │          │
│  Command(resume)│           │                         │          │
│       │         │           │                         │          │
│       ▼         │           ▼                         │          │
│      END        │          END                        │          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Graph Flow

```mermaid
graph TD
    START([🟢 START]) --> T[🔍 triage_node\nAgent 1]
    T -- needs_human = True --> HE[🚨 human_escalate_node\nHITL Path]
    T -- needs_human = False --> RA[🤖 reply_agent_node\nAgent 2]
    HE -- interrupt → human types reply\nCommand resume --> END1([🔴 END])
    RA --> END2([🔴 END])
```

The **compiled graph** is saved as `graph_diagram.png` every time the app starts.

---

## Agents

### Agent 1 — Triage Node (`triage_node`)

| Property | Value |
|---|---|
| **Model** | `gemini-2.0-flash-lite` via `google.genai` client |
| **Output** | Structured JSON (Pydantic `Triage` schema) |
| **Tools** | ❌ None |
| **Purpose** | Classify the incoming message and decide if a human is needed |

**Triage Schema:**

```python
class Triage(BaseModel):
    category:    Literal["billing", "technical", "account", "general"]
    urgency:     Literal["low", "medium", "high"]
    sentiment:   Literal["negative", "neutral", "positive"]
    needs_human: bool   # True → escalate, False → reply_agent
    summary:     str    # One-line summary
```

**Escalation triggers (`needs_human = True`):**
- Refund requests
- Billing disputes
- Cancellations
- Legal / privacy issues
- Angry customers (negative sentiment + billing)

---

### Agent 2 — Reply Agent Node (`reply_agent_node`)

| Property | Value |
|---|---|
| **Model** | `gemini-2.0-flash-lite` via `google_genai` LangChain integration |
| **Agent Type** | ReAct (Reasoning + Acting) — `create_react_agent` from `langgraph.prebuilt` |
| **Tools** | ✅ 6 tools (see [Tools](#tools)) |
| **Purpose** | Fully answer low/medium-urgency questions using real account data |

The agent is built **once at module load** and re-invoked per turn. It uses the `agent_system` prompt from `prompt.yml` which enforces strict rules: always look up the customer first, never guess invoice data, always consult policy RAG before refund decisions.

---

## Conditional Edge — Human in the Loop

The function `route_after_triage()` is the **conditional edge** and is the entire HITL decision point in the graph:

```python
def route_after_triage(state: SupportState) -> str:
    return "human_escalate" if state["needs_human"] else "reply_agent"
```

### How HITL Works (Step by Step)

```
1. Customer sends message
        ↓
2. triage_node classifies it → sets needs_human = True
        ↓
3. route_after_triage() routes to human_escalate_node
        ↓
4. human_escalate_node:
   a. Writes escalation ticket to Postgres `escalations` table
   b. Calls interrupt() → graph is SUSPENDED (state saved in Postgres)
        ↓
5. CLI prompts human operator for reply text
        ↓
6. Human types reply → app.invoke(Command(resume=reply))
        ↓
7. Graph resumes from checkpoint → returns human reply as final reply
        ↓
8. Reply shown to customer
```

**Compile flag:**
```python
graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_escalate"],  # pause BEFORE entering the node
)
```

This means the graph **checkpoints state right before** the escalation node runs, so the human operator can inspect the full triage result before deciding what to say.

---

## Tools

All tools are decorated with `@tool` (LangChain) and available to **Agent 2** only.

### `lookupcustomer(phone_number: str)`
- Looks up a customer by phone number from the `customers` table
- Returns: `customer_id`, `name`, `plan`, `status`, `since`
- **Must be called first** before any account-specific tool
- Returns `"Customer not found"` if not found

### `list_invoices(customer_id: str)`
- Lists all invoices for a given customer ID from the `invoices` table
- Returns: `invoice_id`, `amount`, `invoice_date`, `status`
- Ordered by invoice date (ascending)

### `check_refund_status(invoice_id: str)`
- Checks whether a completed refund exists for a given invoice
- Queries the `refunds` table
- Returns refund details or `{"status": "No refund found"}`

### `refund_policy(query: str)`
- **RAG tool** — searches Qdrant vector store for refund policy passages
- Filters by `doc_type = "refund_policy"`
- Returns top-4 most relevant chunks with similarity scores
- **Must always be called before creating any refund**

### `company_policy(query: str)`
- **RAG tool** — searches Qdrant for general company policy passages
- Filters by `doc_type = "company_policy"`
- Covers: SLA, data retention, acceptable use, account suspension, support response times, privacy
- Returns top-4 most relevant chunks with similarity scores

### `create_refund_request(customer_id, invoice_id, amount, reason, phone_number)`
- Creates a refund request with **smart auto-approval logic**:

| Condition | Result |
|---|---|
| Customer has 2+ invoices in same calendar month | **Auto-approved** (duplicate-charge rule) |
| Otherwise | **Pending** — requires human approval via `approve_refund()` |

- Writes to both `pending_refunds` and `refunds` tables

### `approve_refund(refund_id: str, approver: str)` *(admin tool)*
- Approves a pending refund
- Updates `pending_refunds` status to `approved`
- Inserts completed record into `refunds` table
- Records approver name and timestamp

---

## RAG System

### Embeddings

Uses a custom `GeminiEmbeddings` wrapper class that calls `google.genai` directly:

```python
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = genai_client.models.embed_content(model=self.model, contents=texts)
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> List[float]:
        result = genai_client.models.embed_content(model=self.model, contents=[text])
        return result.embeddings[0].values
```

| Property | Value |
|---|---|
| **Model** | `gemini-embedding-2` |
| **Dimensions** | 3072 |
| **Provider** | Google Gemini (via `google.genai` client) |

> **Why a custom wrapper?** `langchain_google_genai`'s `GoogleGenerativeAIEmbeddings` uses the v1beta REST API which is not accessible with standard Gemini API keys. The custom wrapper uses the `google.genai` SDK directly (same client used for generation).

### Vector Store

| Property | Value |
|---|---|
| **Provider** | Qdrant Cloud |
| **Collection** | `task4` |
| **Integration** | `langchain_qdrant.QdrantVectorStore` |
| **Search type** | `similarity_search_with_score` (cosine similarity) |
| **Top-k** | 4 results per query |

### Document Types (Qdrant Metadata Filter)

| `doc_type` | Used by Tool |
|---|---|
| `refund_policy` | `refund_policy()` |
| `company_policy` | `company_policy()` |

---

## Database Schema

All tables are auto-created on startup if they don't exist.

### `customers` *(pre-existing)*
| Column | Type | Description |
|---|---|---|
| `customer_id` | TEXT | Primary key (e.g. `C-1001`) |
| `name` | TEXT | Customer full name |
| `phone_number` | TEXT | Lookup key |
| `plan` | TEXT | Subscription plan |
| `status` | TEXT | `active` / `suspended` |
| `since` | DATE | Signup date |

### `invoices` *(pre-existing)*
| Column | Type | Description |
|---|---|---|
| `invoice_id` | TEXT | Primary key (e.g. `INV-5013`) |
| `customer_id` | TEXT | Foreign key → customers |
| `amount` | NUMERIC | Invoice amount |
| `invoice_date` | DATE | Billing date |
| `status` | TEXT | `paid` / `unpaid` |

### `refunds` *(auto-created)*
| Column | Type | Description |
|---|---|---|
| `refund_id` | TEXT | Primary key (e.g. `R-a1b2c3d4`) |
| `invoice_id` | TEXT | Related invoice |
| `amount` | NUMERIC | Refund amount |
| `refund_date` | TIMESTAMP | When refund was issued |
| `status` | TEXT | `completed` |

### `pending_refunds` *(auto-created)*
| Column | Type | Description |
|---|---|---|
| `refund_id` | TEXT | Primary key |
| `customer_id` | TEXT | Related customer |
| `invoice_id` | TEXT | Related invoice |
| `amount` | NUMERIC | Refund amount |
| `reason` | TEXT | Reason provided |
| `status` | TEXT | `pending` / `approved` |
| `approver` | TEXT | Who approved it (or `auto-duplicate-rule`) |
| `created_at` | TIMESTAMP | Request creation time |
| `acted_at` | TIMESTAMP | Approval/rejection time |

### `escalations` *(auto-created)*
| Column | Type | Description |
|---|---|---|
| `escalation_id` | TEXT | Primary key (e.g. `E-a1b2c3d4`) |
| `phone_number` | TEXT | Customer's phone (if provided) |
| `message` | TEXT | Original customer message |
| `category` | TEXT | Triage category |
| `urgency` | TEXT | `low` / `medium` / `high` |
| `sentiment` | TEXT | `negative` / `neutral` / `positive` |
| `summary` | TEXT | One-line summary from triage |
| `status` | TEXT | `open` (human queue) |
| `created_at` | TIMESTAMP | When escalation was created |

> The `escalations` table is the **human operator's work queue** — every message that hits the HITL path creates a row here.

---

## Prompts

All prompts are stored in [`prompt.yml`](./prompt.yml) — **zero prompt text lives in Python**.

### `triage_system`
Instructions for Agent 1. Rules:
- Base every field only on the actual message — never invent details
- Set `needs_human = true` for refunds, billing disputes, cancellations, legal/privacy issues, or angry customers
- Prefer `general` + `low` urgency for vague messages

### `reply_system`
Draft reply instructions for the **human escalation path** *(not used in current flow — the human writes the reply directly)*.

### `agent_system`
Instructions for Agent 2 (the ReAct reply agent). Strict rules:
1. Always call `lookupcustomer()` first for any account-specific question
2. Use the returned `customer_id` for invoice tools — never guess
3. Never invent invoice or refund information
4. Always call `refund_policy()` before creating a refund
5. Call `company_policy()` for general company rules (not memory)
6. Customer facts come from tools, never from policy documents
7. Be concise, friendly, and never overpromise outcomes

---

## Token Tracking

Every Gemini API call (triage) is tracked:

```python
_totals = {"input": 0, "output": 0}

def usage_report() -> str:
    # Prints: "1234 in + 567 out tokens = ~$0.0001"
```

Pricing used: `$0.10 / 1M input tokens`, `$0.40 / 1M output tokens` (Flash-Lite). The CLI prints a token report after every turn.

> **Note:** Token tracking only covers the triage node (direct Gemini calls). Tokens used by Agent 2 (LangChain/LangGraph) are not counted in this total.

---

## Project Structure

```
Task5/
├── graph_app.py          # Main application — full LangGraph system
├── prompt.yml            # All system prompts (triage, reply, agent)
├── .env                  # API keys and config (never commit this!)
├── graph_diagram.png     # Auto-generated graph visualization (on startup)
├── readme.md             # This file
└── venv/                 # Python virtual environment
```

---

## Environment Variables

Create a `.env` file in the project root with the following:

```env
# Google Gemini (for LLM + embeddings)
GEMINI_API_KEY=your_gemini_api_key_here

# Qdrant Cloud (for RAG vector store)
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here

# LangSmith (optional — for tracing/observability)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=Task5
```

> **PostgreSQL** is configured directly in code:
> - **Checkpointer DB:** `postgresql://postgres:1234@localhost:5432/fieldwork`
> - **Customer data DB:** `postgresql+psycopg2://postgres:1234@localhost:5432/customer_support`

---

## Prerequisites

| Requirement | Version / Notes |
|---|---|
| Python | 3.10+ |
| PostgreSQL | Running locally on port `5432` |
| `customer_support` database | Must have `customers` and `invoices` tables pre-populated |
| `fieldwork` database | Must exist (for LangGraph checkpointer) |
| Qdrant collection `task4` | Must be indexed with policy documents |

---

## Installation

```bash
# 1. Clone / navigate to the project
cd "Task5"

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install google-genai langchain langgraph langchain-google-genai \
            langchain-qdrant qdrant-client sqlalchemy psycopg2-binary \
            langchain-core pydantic python-dotenv pyyaml

# 4. Set up .env file (see Environment Variables section)

# 5. Ensure PostgreSQL is running with both databases created
```

---

## Running the App

```bash
cd "c:\Users\HP\Desktop\Field Work\Task5"
venv\Scripts\python.exe graph_app.py
```

**On startup the app will:**
1. Auto-create the `pending_refunds`, `refunds`, and `escalations` tables
2. Set up the LangGraph PostgreSQL checkpointer
3. Compile and save the graph diagram to `graph_diagram.png`
4. Start the interactive CLI loop

```
Northstar Support Assistant (LangGraph) — type 'exit' or 'quit' to stop.

You: ▌
```

**To exit:** type `exit` or `quit`.

---

## Example Conversations

### ✅ Path 1 — Automated Reply (needs_human = False)

```
You: What is your support response time?

TRIAGE  : {'category': 'general', 'urgency': 'low', 'sentiment': 'neutral',
           'needs_human': False, 'summary': 'Customer asking about SLA response times'}
ROUTE   : reply_agent

Assistant
According to our company policy, Northstar Services responds to support
requests within 24 hours for standard plans and 4 hours for enterprise plans.
You can reach us any time via our support portal!

================================================================================
TOKENS  : 312 in + 48 out tokens = ~$0.0001
```

---

### 🚨 Path 2 — Human Escalation (needs_human = True)

```
You: I was charged twice this month and I want a refund immediately!

TRIAGE  : {'category': 'billing', 'urgency': 'high', 'sentiment': 'negative',
           'needs_human': True, 'summary': 'Customer reports duplicate charge and demands refund'}
ROUTE   : human_escalate (HITL)

⚠  Human escalation required.
   Summary : Customer reports duplicate charge and demands refund
   Category: billing | Urgency: high

Human operator reply (or press Enter to skip): Hi! I'm so sorry about the
duplicate charge. I've reviewed your account and I'll process a full refund
within 2-3 business days. You'll receive a confirmation email shortly.

Assistant
Hi! I'm so sorry about the duplicate charge. I've reviewed your account
and I'll process a full refund within 2-3 business days. You'll receive
a confirmation email shortly.

================================================================================
TOKENS  : 287 in + 52 out tokens = ~$0.0001
```

---

### 🔍 Path 3 — Account Lookup + Invoice Query

```
You: Can you check my invoices? My phone is 555-0102.

TRIAGE  : {'category': 'billing', 'urgency': 'low', 'sentiment': 'neutral',
           'needs_human': False, 'summary': 'Customer wants to see their invoices'}
ROUTE   : reply_agent

Assistant
I found your account — you're Jane Smith on the Pro plan (active since 2023).
You have 3 invoices:
• INV-5011 — $99.00 — 2024-01-15 — paid
• INV-5012 — $99.00 — 2024-02-15 — paid
• INV-5013 — $99.00 — 2024-03-15 — unpaid

================================================================================
```

---

## Key Design Decisions

### 1. Only Two Agent Nodes
The graph has exactly **two AI agents** (`triage_node` and `reply_agent_node`). The `human_escalate_node` is **not an AI agent** — it is a pure Python node that logs to the database and suspends the graph.

### 2. Conditional Edge = HITL Decision
The entire Human-in-the-Loop mechanism is expressed as a **LangGraph conditional edge**, not as an `if` statement buried inside a node. This makes the escalation decision visible in the graph visualization and traceable in LangSmith.

### 3. Structured Output for Triage
Agent 1 uses **constrained JSON generation** (`response_mime_type: application/json` + `response_schema: Triage`) instead of free-text parsing. This eliminates hallucinated routing decisions — the `needs_human` field is always a clean boolean.

### 4. PostgreSQL Checkpointing
LangGraph's `PostgresSaver` checkpointer saves the **full graph state** after every node. This means:
- Escalated conversations are **durably suspended** — the server can restart and the conversation resumes from where it left off
- Each `thread_id` is an independent conversation session

### 5. Prompt Externalization
Zero prompt text lives in Python code. All three system prompts (`triage_system`, `reply_system`, `agent_system`) are loaded from `prompt.yml` at startup. Changing prompt behavior requires editing only the YAML file.

### 6. Custom Embedding Wrapper
`langchain_google_genai.GoogleGenerativeAIEmbeddings` uses the v1beta REST API which is inaccessible with standard Gemini API keys. The custom `GeminiEmbeddings` class wraps `google.genai` directly (the same client used for generation), bypassing this limitation.

---

## Dependencies

| Package | Purpose |
|---|---|
| `google-genai` | Gemini LLM + embeddings (triage node, GeminiEmbeddings) |
| `langchain` | Tool decorators, `init_chat_model`, core abstractions |
| `langgraph` | StateGraph, conditional edges, `interrupt()`, `PostgresSaver` |
| `langchain-core` | Base `Embeddings` class for the custom wrapper |
| `langchain-qdrant` | `QdrantVectorStore` integration |
| `qdrant-client` | Qdrant Cloud client + filter models |
| `sqlalchemy` | PostgreSQL ORM / query engine |
| `psycopg2-binary` | PostgreSQL driver |
| `pydantic` | Triage schema validation |
| `python-dotenv` | `.env` file loading |
| `pyyaml` | `prompt.yml` loading |

---

## Graph Visualization

The compiled graph is automatically saved as `graph_diagram.png` on every startup:

```python
def save_graph_diagram(app, path: str = "graph_diagram.png") -> None:
    png_bytes = app.get_graph().draw_mermaid_png()
    Path(path).write_bytes(png_bytes)
```

This uses LangGraph's built-in Mermaid renderer (requires internet access). If offline, the Mermaid source text is printed to the console instead.

---

*Built with ❤️ using LangGraph + Google Gemini — Task 5, Field Work Project*
