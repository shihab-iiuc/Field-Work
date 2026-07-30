
# --- 0. Setup ------------------------------------------------------------
import os
import uuid
from pathlib import Path
from typing import Annotated, Optional, TypedDict

import yaml
from dotenv import load_dotenv

load_dotenv(override=True)   # reads .env — GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY


# --- 1. Prompts  (ALL come from prompt.yml — nothing hard-coded here) ----
_PROMPTS = yaml.safe_load((Path(__file__).parent / "prompt.yml").read_text(encoding="utf-8"))
TRIAGE_SYSTEM = _PROMPTS["triage_system"]
REPLY_SYSTEM  = _PROMPTS["reply_system"]
AGENT_SYSTEM  = _PROMPTS["agent_system"]


# --- 2. Structured triage schema -----------------------------------------
from typing import Literal
from pydantic import BaseModel, Field


class Triage(BaseModel):
    category:    Literal["billing", "technical", "account", "general"] = Field(
        description="The main topic of the customer's message."
    )
    urgency:     Literal["low", "medium", "high"] = Field(
        description="How quickly this needs a response."
    )
    sentiment:   Literal["negative", "neutral", "positive"] = Field(
        description="The customer's mood in the message."
    )
    needs_human: bool = Field(
        description="True if this must be escalated to a human — refunds, billing disputes, "
                    "cancellations, legal/privacy issues, or an angry customer."
    )
    summary:     str = Field(description="One-line summary of what the customer wants.")


# --- 3. Gemini client for structured triage calls ------------------------
from google import genai

genai_client = genai.Client()           # reads GEMINI_API_KEY from the environment
TRIAGE_MODEL  = "gemini-2.0-flash-lite"  # correct model name

_totals = {"input": 0, "output": 0}


def track(usage) -> None:
    _totals["input"]  += usage.prompt_token_count     or 0
    _totals["output"] += usage.candidates_token_count or 0


def usage_report() -> str:
    IN_PRICE, OUT_PRICE = 0.10, 0.40      # $/1M tokens for Flash-Lite (check current pricing)
    cost = _totals["input"] / 1e6 * IN_PRICE + _totals["output"] / 1e6 * OUT_PRICE
    return f"{_totals['input']} in + {_totals['output']} out tokens = ~${cost:.4f}"


# --- 4. RAG store (Qdrant) -----------------------------------------------
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings
from typing import List


class GeminiEmbeddings(Embeddings):
    """Thin LangChain-compatible wrapper around google.genai embed_content.
    Uses the same `genai_client` already instantiated above — avoids the
    broken v1beta path that langchain_google_genai uses."""

    def __init__(self, model: str = "gemini-embedding-2"):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = genai_client.models.embed_content(model=self.model, contents=texts)
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> List[float]:
        result = genai_client.models.embed_content(model=self.model, contents=[text])
        return result.embeddings[0].values


qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
embeddings   = GeminiEmbeddings(model="gemini-embedding-2")
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="task4",
    embedding=embeddings,
)


# --- 5. Chat model for the reply_agent -----------------------------------
from langchain.chat_models import init_chat_model

AGENT_MODEL = "google_genai:gemini-2.0-flash-lite"   # correct model prefix + name
chat_model  = init_chat_model(AGENT_MODEL)


# --- 6. Postgres: customers / invoices + refunds + escalations -----------
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://postgres:1234@localhost:5432/customer_support"
engine = create_engine(DATABASE_URL)

with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS pending_refunds (
            refund_id   TEXT PRIMARY KEY,
            customer_id TEXT,
            invoice_id  TEXT,
            amount      NUMERIC,
            reason      TEXT,
            status      TEXT,
            approver    TEXT,
            created_at  TIMESTAMP,
            acted_at    TIMESTAMP
        )
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS refunds (
            refund_id   TEXT PRIMARY KEY,
            invoice_id  TEXT,
            amount      NUMERIC,
            refund_date TIMESTAMP,
            status      TEXT
        )
    """))
    # Every message the conditional edge routes to "human_escalate" gets a row
    # here, giving a human teammate a queue to work from.
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS escalations (
            escalation_id TEXT PRIMARY KEY,
            phone_number  TEXT,
            message       TEXT,
            category      TEXT,
            urgency       TEXT,
            sentiment     TEXT,
            summary       TEXT,
            status        TEXT,
            created_at    TIMESTAMP
        )
    """))


# --- 7. Tools ------------------------------------------------------------
from langchain.tools import tool


@tool
def lookupcustomer(phone_number: str):
    """Look up a Northstar customer by phone number. Returns customer_id, name,
    plan, status, and signup date. Call this FIRST when you need the account."""
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT customer_id, name, plan, status, since
                FROM customers WHERE phone_number = :phone_number
            """),
            {"phone_number": phone_number},
        ).mappings().first()
    return dict(result) if result else "Customer not found"


@tool
def list_invoices(customer_id: str):
    """List every invoice for a customer_id (e.g. 'C-1001'). Call lookupcustomer
    first to get the customer_id."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT invoice_id, amount, invoice_date, status
                FROM invoices WHERE customer_id=:customer_id
                ORDER BY invoice_date
            """),
            {"customer_id": customer_id},
        ).mappings().all()
    return [dict(r) for r in rows]


@tool
def check_refund_status(invoice_id: str):
    """Check whether a refund exists for an invoice id (e.g. 'INV-5013')."""
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT refund_id, amount, refund_date, status
                FROM refunds WHERE invoice_id=:invoice_id
            """),
            {"invoice_id": invoice_id},
        ).mappings().first()
    return dict(row) if row else {"invoice_id": invoice_id, "status": "No refund found"}


@tool
def refund_policy(query: str) -> str:
    """Search Northstar's Refund Policy. ALWAYS call before creating/approving
    a refund — eligibility windows, duplicate charges, non-refundable items,
    approval limits, processing time."""
    hits = vector_store.similarity_search_with_score(
        query, k=4,
        filter=Filter(must=[FieldCondition(key="metadata.doc_type",
                                           match=MatchValue(value="refund_policy"))]),
    )
    if not hits:
        return "No relevant refund policy passage found."
    return "\n\n".join(f"(score={score:.3f}) {doc.page_content}" for doc, score in hits)


@tool
def company_policy(query: str) -> str:
    """Search Northstar's general company policies: SLA, data retention,
    acceptable use, account suspension, support response times, plan
    changes, privacy."""
    hits = vector_store.similarity_search_with_score(
        query, k=4,
        filter=Filter(must=[FieldCondition(key="metadata.doc_type",
                                           match=MatchValue(value="company_policy"))]),
    )
    if not hits:
        return "No relevant company policy passage found."
    return "\n\n".join(f"(score={score:.3f}) {doc.page_content}" for doc, score in hits)


@tool
def create_refund_request(customer_id: str = None, invoice_id: str = None,
                           amount: float = None, reason: str = "",
                           phone_number: str = None):
    """Create a refund request. Auto-approves if the customer has 2+ charges
    in the same calendar month (duplicate-charge rule); otherwise creates a
    pending_refunds row for human approval via approve_refund()."""
    if phone_number:
        cust = lookupcustomer.invoke({"phone_number": phone_number})
        if cust == "Customer not found":
            return {"status": "not_a_customer", "message": "Customer not found for phone number."}
        customer_id = cust.get("customer_id")

    if not customer_id or not invoice_id:
        return {"status": "error",
                "message": "customer_id and invoice_id are required (or provide phone_number)."}

    with engine.begin() as conn:
        inv = conn.execute(
            text("SELECT invoice_id, amount, invoice_date FROM invoices "
                 "WHERE invoice_id=:invoice_id AND customer_id=:customer_id"),
            {"invoice_id": invoice_id, "customer_id": customer_id},
        ).mappings().first()
        if not inv:
            return {"status": "invoice_not_found", "invoice_id": invoice_id}

        invoice_date = inv["invoice_date"]
        cnt = conn.execute(
            text("""SELECT count(*) FROM invoices WHERE customer_id=:customer_id
                     AND extract(year  from invoice_date)=extract(year  from :invoice_date)
                     AND extract(month from invoice_date)=extract(month from :invoice_date)"""),
            {"customer_id": customer_id, "invoice_date": invoice_date},
        ).scalar()

        refund_id = "R-" + uuid.uuid4().hex[:8]

        if cnt and int(cnt) >= 2:
            conn.execute(
                text("INSERT INTO refunds(refund_id, invoice_id, amount, refund_date, status) "
                     "VALUES (:refund_id, :invoice_id, :amount, now(), 'completed')"),
                {"refund_id": refund_id, "invoice_id": invoice_id, "amount": inv["amount"]},
            )
            conn.execute(
                text("""INSERT INTO pending_refunds
                        (refund_id, customer_id, invoice_id, amount, reason,
                         status, created_at, acted_at, approver)
                        VALUES (:refund_id, :customer_id, :invoice_id, :amount, :reason,
                                'approved', now(), now(), 'auto-duplicate-rule')"""),
                {"refund_id": refund_id, "customer_id": customer_id,
                 "invoice_id": invoice_id, "amount": inv["amount"], "reason": reason},
            )
            return {"refund_id": refund_id, "status": "completed",
                    "message": "Auto-approved due to duplicate monthly charge rule."}

        conn.execute(
            text("""INSERT INTO pending_refunds
                    (refund_id, customer_id, invoice_id, amount, reason, status, created_at)
                    VALUES (:refund_id, :customer_id, :invoice_id, :amount, :reason, 'pending', now())"""),
            {"refund_id": refund_id, "customer_id": customer_id,
             "invoice_id": invoice_id, "amount": amount or inv["amount"], "reason": reason},
        )
    return {"refund_id": refund_id, "status": "pending",
            "message": "Refund request created. Human approval required."}


@tool
def approve_refund(refund_id: str, approver: str):
    """Human approves the refund; executes it and records it in `refunds`."""
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT * FROM pending_refunds WHERE refund_id=:refund_id"),
            {"refund_id": refund_id},
        ).mappings().first()
        if not row:
            return {"status": "not_found", "refund_id": refund_id}
        if row["status"] != "pending":
            return {"status": row["status"], "message": "Already processed"}
        conn.execute(
            text("INSERT INTO refunds(refund_id, invoice_id, amount, refund_date, status) "
                 "VALUES (:refund_id, :invoice_id, :amount, now(), 'completed')"),
            {"refund_id": refund_id, "invoice_id": row["invoice_id"], "amount": row["amount"]},
        )
        conn.execute(
            text("UPDATE pending_refunds SET status='approved', approver=:approver, "
                 "acted_at=now() WHERE refund_id=:refund_id"),
            {"approver": approver, "refund_id": refund_id},
        )
    return {"status": "approved", "refund_id": refund_id}


# --- 8. Graph state -------------------------------------------------------
class SupportState(TypedDict, total=False):
    customer_message: str
    phone_number:     Optional[str]
    triage:           dict    # Triage.model_dump()
    needs_human:      bool
    reply:            str


# =========================================================================
# --- 9. Nodes
#
#  Graph topology (only TWO agent nodes + one conditional edge):
#
#   START
#     │
#     ▼
#  [triage_node]          ← Agent 1: structured-output Gemini call
#     │
#     ├── needs_human=True  ──►  [human_escalate_node]  ──► END
#     │                          (logs ticket + interrupt for HITL)
#     │
#     └── needs_human=False ──►  [reply_agent_node]  ──► END
#                                (Agent 2: ReAct tool-calling agent)
#
# The conditional edge IS the "human in the loop" decision point.
# =========================================================================

def triage_node(state: SupportState) -> dict:
    """Agent 1 — structured-output call (no tools). Classifies the message
    and sets `needs_human`, which the conditional edge reads to route the
    conversation either to the human-escalate path or to the reply agent."""
    resp = genai_client.models.generate_content(
        model=TRIAGE_MODEL,
        contents=state["customer_message"],
        config={
            "system_instruction": TRIAGE_SYSTEM,
            "response_mime_type": "application/json",
            "response_schema": Triage,
        },
    )
    track(resp.usage_metadata)
    t: Triage = resp.parsed
    return {"triage": t.model_dump(), "needs_human": t.needs_human}


def human_escalate_node(state: SupportState) -> dict:
    """Human-in-the-loop path — reached only when triage sets needs_human=True.

    1. Logs a ticket row to `escalations` so a human teammate has a queue.
    2. Uses LangGraph's `interrupt()` to pause execution and hand control
       back to the caller (the human operator / front-end).  The caller can
       resume the graph later with a human-written reply via
       `app.invoke(Command(resume=human_reply), config=config)`.

    If you prefer a fire-and-forget model (no resume), simply remove the
    interrupt() call — the node will still log the ticket and the graph
    will end with an empty `reply`, which you can detect in the CLI loop.
    """
    from langgraph.types import interrupt   # LangGraph ≥ 0.2 HITL primitive

    t = state["triage"]

    # 1. Persist the escalation ticket
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO escalations
                    (escalation_id, phone_number, message, category,
                     urgency, sentiment, summary, status, created_at)
                    VALUES (:eid, :phone, :msg, :cat, :urg, :sent, :summary, 'open', now())"""),
            {
                "eid":     "E-" + uuid.uuid4().hex[:8],
                "phone":   state.get("phone_number"),
                "msg":     state["customer_message"],
                "cat":     t["category"],
                "urg":     t["urgency"],
                "sent":    t["sentiment"],
                "summary": t["summary"],
            },
        )

    # 2. Pause graph and ask a human operator for the reply text.
    #    The value returned by interrupt() is whatever the operator sends back
    #    when they resume the thread.  Until then, the graph is suspended.
    human_reply = interrupt(
        {
            "reason":   "Human escalation required",
            "summary":  t["summary"],
            "category": t["category"],
            "urgency":  t["urgency"],
            "message":  state["customer_message"],
        }
    )

    # 3. Use the human-provided text as the final reply
    return {"reply": human_reply or "(Escalated to a human agent — response pending.)"}


# --- Reply agent (Agent 2) — built once at module load -------------------
from langgraph.prebuilt import create_react_agent

_reply_agent_tools = [
    lookupcustomer,
    list_invoices,
    check_refund_status,
    refund_policy,
    company_policy,
    create_refund_request,
]

reply_agent = create_react_agent(
    model=chat_model,
    tools=_reply_agent_tools,
    prompt=AGENT_SYSTEM,
)


def reply_agent_node(state: SupportState) -> dict:
    """Agent 2 — ReAct tool-calling agent for everything triage did NOT flag
    as needing a human.  Free to call account/policy tools and actually answer
    the customer's question."""
    result = reply_agent.invoke(
        {"messages": [{"role": "user", "content": state["customer_message"]}]}
    )
    last = result["messages"][-1].content
    if isinstance(last, list):
        text_out = "\n".join(
            b["text"] for b in last if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        text_out = last
    return {"reply": text_out}


# --- Conditional edge router ---------------------------------------------
def route_after_triage(state: SupportState) -> str:
    """THE conditional edge — reads `needs_human` set by the triage node and
    routes to either the human-escalation path or the automated reply agent.
    This single function IS the human-in-the-loop decision point in the graph."""
    return "human_escalate" if state["needs_human"] else "reply_agent"


# --- 10. Build the graph -------------------------------------------------
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:1234@localhost:5432/fieldwork?sslmode=disable"


def build_graph(checkpointer):
    """
    Two-agent graph with a conditional edge for human-in-the-loop:

        START → triage ──(needs_human=True)──► human_escalate → END
                        └─(needs_human=False)─► reply_agent   → END
    """
    graph = StateGraph(SupportState)

    # Nodes — exactly two agents
    graph.add_node("triage",          triage_node)
    graph.add_node("reply_agent",     reply_agent_node)
    graph.add_node("human_escalate",  human_escalate_node)

    # Entry point
    graph.add_edge(START, "triage")

    # Conditional edge (the human-in-the-loop decision)
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "human_escalate": "human_escalate",   # needs_human = True
            "reply_agent":    "reply_agent",       # needs_human = False
        },
    )

    # Both paths converge at END
    graph.add_edge("human_escalate", END)
    graph.add_edge("reply_agent",    END)

    # interrupt_before lets the checkpointer suspend the graph at
    # human_escalate_node so a human can inspect and resume it.
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_escalate"],   # pause BEFORE entering the node
    )


def save_graph_diagram(app, path: str = "graph_diagram.png") -> None:
    """Renders the compiled graph to a PNG via LangGraph's Mermaid renderer."""
    try:
        png_bytes = app.get_graph().draw_mermaid_png()
        Path(path).write_bytes(png_bytes)
        print(f"Graph diagram saved to {path}")
    except Exception as e:
        print(f"Could not render diagram ({e}). Mermaid source:")
        print(app.get_graph().draw_mermaid())


# --- 11. CLI loop --------------------------------------------------------
def chat(app, config, question: str) -> None:
    question = question.strip()
    if not question:
        print("Please enter a message.")
        return

    try:
        # First invoke — runs triage and either auto-replies or suspends for HITL
        result = app.invoke({"customer_message": question}, config=config)

        triage_data = result.get("triage", {})
        needs_human = result.get("needs_human", False)

        print("\nTRIAGE  :", triage_data)
        print("ROUTE   :", "human_escalate (HITL)" if needs_human else "reply_agent")

        if needs_human:
            # The graph is now suspended at human_escalate_node.
            # A real front-end would queue this to a human operator.
            # In the CLI we prompt the operator inline.
            print("\n⚠  Human escalation required.")
            print("   Summary :", triage_data.get("summary"))
            print("   Category:", triage_data.get("category"),
                  "| Urgency:", triage_data.get("urgency"))
            human_reply = input("\nHuman operator reply (or press Enter to skip): ").strip()
            if not human_reply:
                human_reply = "(Escalated to a human agent — response pending.)"

            # Resume the suspended graph with the human-provided reply
            from langgraph.types import Command
            result = app.invoke(Command(resume=human_reply), config=config)

        print("\nAssistant\n")
        print(result.get("reply", "(no reply generated)"))
        print("\n" + "=" * 80)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        app = build_graph(checkpointer)
        save_graph_diagram(app)

        config = {"configurable": {"thread_id": "customer-session-1"}}
        print("Northstar Support Assistant (LangGraph) — type 'exit' or 'quit' to stop.\n")

        while True:
            user_input = input("You: ")
            if user_input.strip().lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            chat(app, config, user_input)
            print("TOKENS  :", usage_report())
