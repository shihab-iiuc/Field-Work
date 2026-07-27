import os
from dotenv import load_dotenv
load_dotenv(override=True)  # reads .env — GEMINI_API_KEY
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

from langchain_community.document_loaders import PyPDFLoader
from qdrant_client.models import Filter, FieldCondition, MatchValue
from uuid import NAMESPACE_URL, uuid5


# RAG setup: single existing Qdrant collection ("task4", 3072-dim).


REFUND_POLICY_PATH = r"C:\Users\HP\Desktop\Field Work\Task4\refund-policy.pdf"
COMPANY_POLICY_PATH = r"C:\Users\HP\Desktop\Field Work\Task4\company-policy.pdf"

collection_name = "task4"

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Load + split + tag + index the refund policy
refund_docs = PyPDFLoader(REFUND_POLICY_PATH).load()
refund_splits = splitter.split_documents(refund_docs)
for d in refund_splits:
    d.metadata["doc_type"] = "refund_policy"
refund_ids = [str(uuid5(NAMESPACE_URL, f"{d.metadata}|{d.page_content}")) for d in refund_splits]
vector_store.add_documents(refund_splits, ids=refund_ids)

# Load + split + tag + index the company policy
company_docs = PyPDFLoader(COMPANY_POLICY_PATH).load()
company_splits = splitter.split_documents(company_docs)
for d in company_splits:
    d.metadata["doc_type"] = "company_policy"
company_ids = [str(uuid5(NAMESPACE_URL, f"{d.metadata}|{d.page_content}")) for d in company_splits]
vector_store.add_documents(company_splits, ids=company_ids)

# Stable IDs (uuid5 of metadata+content) mean rerunning this script just
# overwrites the same points instead of duplicating them.

# Model Initialization
from langchain.chat_models import init_chat_model
MODEL = "google_genai:gemini-3.1-flash-lite"   # ":" — this string is the swap point
model = init_chat_model(MODEL)

# Dummy Customer database
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://postgres:1234@localhost:5432/customer_support"
engine = create_engine(DATABASE_URL)

# Ensure pending_refunds and refunds tables exist
with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS pending_refunds (
            refund_id TEXT PRIMARY KEY,
            customer_id TEXT,
            invoice_id TEXT,
            amount NUMERIC,
            reason TEXT,
            status TEXT,
            approver TEXT,
            created_at TIMESTAMP,
            acted_at TIMESTAMP
        )
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS refunds (
            refund_id TEXT PRIMARY KEY,
            invoice_id TEXT,
            amount NUMERIC,
            refund_date TIMESTAMP,
            status TEXT
        )
    """))

# all tools
from langchain.tools import tool


@tool
def lookupcustomer(phone_number: str):
    """
    Look up a Northstar customer by their phone number.

    Returns their customer_id, name, plan, account status, and signup date.

    Use this FIRST when a customer message includes a phone number and you need
    their account.
    """

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT
                    customer_id,
                    name,
                    plan,
                    status,
                    since
                FROM customers
                WHERE phone_number = :phone_number
            """),
            {"phone_number": phone_number}
        ).mappings().first()

    if result:
        return dict(result)

    return "Customer not found"


@tool
def list_invoices(customer_id: str):
    """List every invoice for a customer_id (e.g. 'C-1001'): invoice id, month, amount, status.
    Call lookup_customer first to get the customer_id.
    """

    with engine.connect() as conn:

        rows = conn.execute(
            text("""
                SELECT
                    invoice_id,
                    amount,
                    invoice_date,
                    status
                FROM invoices
                WHERE customer_id=:customer_id
                ORDER BY invoice_date
            """),
            {"customer_id": customer_id}
        ).mappings().all()

    return [dict(r) for r in rows]


@tool
def check_refund_status(invoice_id: str):
    """Check whether a refund exists for an invoice id (e.g. 'INV-5013'), and its status and refund_date.
    Use when a customer asks 'where is my refund?'.
    """

    with engine.connect() as conn:

        row = conn.execute(
            text("""
                SELECT
                    refund_id,
                    amount,
                    refund_date,
                    status
                FROM refunds
                WHERE invoice_id=:invoice_id
            """),
            {"invoice_id": invoice_id}
        ).mappings().first()

    if row:
        return dict(row)

    return {
        "invoice_id": invoice_id,
        "status": "No refund found"
    }


# --- Refund policy (RAG, backed by refund-policy.pdf in Qdrant) ---
@tool
def refund_policy(query: str) -> str:
    """Search Northstar's Refund Policy. ALWAYS call this before creating or
    approving a refund, e.g. for questions about eligibility windows,
    duplicate charges, non-refundable items, approval limits, or processing time.
    """
    hits = vector_store.similarity_search_with_score(
        query,
        k=4,
        filter=Filter(must=[FieldCondition(key="metadata.doc_type", match=MatchValue(value="refund_policy"))]),
    )
    if not hits:
        return "No relevant refund policy passage found."

    return "\n\n".join(f"(score={score:.3f}) {doc.page_content}" for doc, score in hits)


# --- Company policy (RAG, backed by company-policy.pdf in Qdrant) ---
@tool
def company_policy(query: str) -> str:
    """Search and retrieve information about Northstar's general company
    policies including SLA, data retention, acceptable use, account
    suspension, support response times, plan changes, and privacy.
    Use this tool when a customer asks about company rules or terms.
    """
    hits = vector_store.similarity_search_with_score(
        query,
        k=4,
        filter=Filter(must=[FieldCondition(key="metadata.doc_type", match=MatchValue(value="company_policy"))]),
    )
    if not hits:
        return "No relevant company policy passage found."

    return "\n\n".join(f"(score={score:.3f}) {doc.page_content}" for doc, score in hits)


# --- Create refund request (human-in-the-loop, but auto-approve duplicates) ---
@tool
def create_refund_request(customer_id: str = None, invoice_id: str = None, amount: float = None, reason: str = "", phone_number: str = None):
    """Create a refund request and mark it pending for human approval.

    Behavior:
    - Call `refund_policy` FIRST to confirm the request is plausibly eligible.
    - Verifies the customer via `lookupcustomer` if `phone_number` or 'name' is provided.
    - Validates the invoice exists and belongs to the customer.
    - If the customer has 2+ charges in the same calendar month, auto-approve and execute the refund immediately.
    - Otherwise create a pending_refunds row and require human approval via `approve_refund`.

    Returns a refund request id and status.
    """
    # Verify customer via phone if provided
    if phone_number:
        cust = lookupcustomer(phone_number)
        if cust == "Customer not found":
            return {"status": "not_a_customer", "message": "Customer not found for phone number."}
        customer_id = cust.get("customer_id")

    if not customer_id or not invoice_id:
        return {"status": "error", "message": "customer_id and invoice_id are required (or provide phone_number)."}

    with engine.begin() as conn:
        inv = conn.execute(text("SELECT invoice_id, amount, invoice_date FROM invoices WHERE invoice_id=:invoice_id AND customer_id=:customer_id"), {"invoice_id": invoice_id, "customer_id": customer_id}).mappings().first()
        if not inv:
            return {"status": "invoice_not_found", "invoice_id": invoice_id}

        invoice_date = inv["invoice_date"]

        # Count invoices for the same customer in the same calendar month
        cnt = conn.execute(
            text("SELECT count(*) FROM invoices WHERE customer_id=:customer_id AND extract(year from invoice_date)=extract(year from :invoice_date) AND extract(month from invoice_date)=extract(month from :invoice_date)"),
            {"customer_id": customer_id, "invoice_date": invoice_date}
        ).scalar()

        import uuid
        refund_id = "R-" + uuid.uuid4().hex[:8]

        if cnt and int(cnt) >= 2:
            # Auto-approve duplicate-charge refund
            conn.execute(
                text("INSERT INTO refunds(refund_id, invoice_id, amount, refund_date, status) VALUES (:refund_id, :invoice_id, :amount, now(), 'completed')"),
                {"refund_id": refund_id, "invoice_id": invoice_id, "amount": inv["amount"]}
            )
            conn.execute(
                text("INSERT INTO pending_refunds(refund_id, customer_id, invoice_id, amount, reason, status, created_at, acted_at, approver) VALUES (:refund_id, :customer_id, :invoice_id, :amount, :reason, 'approved', now(), now(), 'auto-duplicate-rule')"),
                {"refund_id": refund_id, "customer_id": customer_id, "invoice_id": invoice_id, "amount": inv["amount"], "reason": reason}
            )

            return {"refund_id": refund_id, "status": "completed", "message": "Auto-approved due to duplicate monthly charge rule."}

        # Otherwise create pending request for human approval
        conn.execute(
            text("INSERT INTO pending_refunds(refund_id, customer_id, invoice_id, amount, reason, status, created_at) VALUES (:refund_id, :customer_id, :invoice_id, :amount, :reason, 'pending', now())"),
            {"refund_id": refund_id, "customer_id": customer_id, "invoice_id": invoice_id, "amount": amount or inv["amount"], "reason": reason}
        )

    return {"refund_id": refund_id, "status": "pending", "message": "Refund request created. Human approval required."}


# --- Approve and execute refund (human action) ---
@tool
def approve_refund(refund_id: str, approver: str):
    """Human approves the refund; this executes the refund and records it in `refunds` table."""
    with engine.begin() as conn:
        row = conn.execute(text("SELECT * FROM pending_refunds WHERE refund_id=:refund_id"), {"refund_id": refund_id}).mappings().first()
        if not row:
            return {"status": "not_found", "refund_id": refund_id}
        if row["status"] != "pending":
            return {"status": row["status"], "message": "Already processed"}

        # Insert into refunds table and mark pending as approved
        conn.execute(
            text("INSERT INTO refunds(refund_id, invoice_id, amount, refund_date, status) VALUES (:refund_id, :invoice_id, :amount, now(), 'completed')"),
            {"refund_id": refund_id, "invoice_id": row["invoice_id"], "amount": row["amount"]}
        )

        conn.execute(text("UPDATE pending_refunds SET status='approved', approver=:approver, acted_at=now() WHERE refund_id=:refund_id"), {"approver": approver, "refund_id": refund_id})

    return {"status": "approved", "refund_id": refund_id}


# continues agent
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:1234@localhost:5432/fieldwork?sslmode=disable"

SUPPORT_SYSTEM = """
You are a customer support assistant for Northstar Services.

Rules:
1. Always identify the customer using lookupcustomer().
2. Use customer_id from lookupcustomer() before calling invoice tools.
3. Never guess invoice or refund information.
4. Before creating or approving ANY refund, call refund_policy(query) to
   retrieve the relevant refund-policy section and check the request
   against it. Do not rely on memory for refund rules.
5. When a customer asks about general company rules or terms (support
   hours, response times, data/privacy, account security, acceptable
   conduct, service incidents, etc.), call company_policy(query) to
   retrieve the relevant section instead of guessing.
6. Customer-specific facts (plan, invoice status, refund status) must
   always come from the account-system tools, not from policy documents.
7. Be concise and friendly.
"""


def chat(agent, config, question: str):

    question = question.strip()

    if not question:
        print("Please enter a message.")
        return

    try:

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            },
            config=config,
        )

        last_message = result["messages"][-1].content

        print("\nAssistant\n")

        if isinstance(last_message, list):

            for block in last_message:

                if isinstance(block, dict):

                    if block.get("type") == "text":
                        print(block["text"])

                else:
                    print(block)

        elif isinstance(last_message, str):
            print(last_message)

        else:
            print(last_message)

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":

    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:

        checkpointer.setup()

        agent = create_agent(
            model=model,
            tools=[
                lookupcustomer,
                list_invoices,
                check_refund_status,
                refund_policy,
                company_policy,
                create_refund_request,
                approve_refund,
            ],
            checkpointer=checkpointer,
            system_prompt=SUPPORT_SYSTEM,
        )

        config = {
            "configurable": {
                "thread_id": "customer-session-2"
            }
        }

        print("Northstar Support Assistant — type 'exit' or 'quit' to stop.\n")

        while True:
            user_input = input("You: ")

            if user_input.strip().lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            chat(agent, config, user_input)