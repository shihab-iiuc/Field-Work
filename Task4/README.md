# Northstar Support Assistant (AI Customer Support Agent)

An intelligent, stateful AI customer support agent for **Northstar Services** built using **LangChain**, **LangGraph**, **Google Gemini**, **Qdrant Vector Database**, and **PostgreSQL**.

---

## 📌 Overview (What It Actually Does)

The **Northstar Support Assistant** automates tier-1 customer support inquiries with context-aware RAG (Retrieval-Augmented Generation) and relational database integration.

### Core Functions:
1. **Customer Account & Invoice Lookup**: Authenticates customers by phone number and retrieves account status, plan information, and historical invoice records.
2. **Policy Retrieval via RAG**: Answers queries about company policies, SLAs, privacy, and refund eligibility using document embeddings stored in a **Qdrant** vector database (`refund-policy.pdf` and `company-policy.pdf`).
3. **Automated & Human-in-the-Loop Refund Processing**:
   - Checks eligibility rules against policy documents.
   - **Auto-approves** refunds if duplicate billing occurs within the same calendar month.
   - Creates a `pending` refund request for standard cases requiring manual human supervisor approval.
4. **Persistent Chat Sessions**: Uses **LangGraph PostgreSQL Checkpointer** (`PostgresSaver`) to retain conversation history across turns and user sessions.

---

## ✨ Key Features

- 🔍 **Phone-Based Customer Identification**: Automatically queries PostgreSQL to pull account details before taking actions.
- 📑 **Dual Document RAG Search**:
  - `refund_policy`: Searches `refund-policy.pdf` for refund rules, eligibility windows, non-refundable items, and limits.
  - `company_policy`: Searches `company-policy.pdf` for SLAs, acceptable use, data retention, and terms of service.
- 💳 **Smart Refund Workflow**:
  - Auto-detects multiple charges in a calendar month and automatically issues a refund (`status: completed`).
  - Flags non-duplicate refund requests into a `pending_refunds` table for human sign-off (`approve_refund`).
- 🧠 **Gemini LLM Integration**: Driven by `gemini-3.1-flash-lite` and `gemini-embedding-2` embeddings.
- 🔄 **Stateful Multi-Turn Conversations**: State persistence using LangGraph PostgreSQL checkpointer (`thread_id` management).

---

## 🛠️ Tech Stack & Dependencies

- **Language**: Python 3.13+
- **LLM & Embeddings**: Google Generative AI (`gemini-3.1-flash-lite`, `models/gemini-embedding-2`)
- **Agent Framework**: LangChain & LangGraph (`langchain`, `langgraph-checkpoint-postgres`)
- **Vector Database**: Qdrant (`langchain-qdrant`, `qdrant-client`)
- **Relational Database**: PostgreSQL + SQLAlchemy (`sqlalchemy`, `psycopg2`)
- **Document Loading & Chunking**: PyPDF Loader (`langchain-community`, `pypdf`, `RecursiveCharacterTextSplitter`)
- **Observability (Optional)**: LangSmith tracing

---

## 📁 Project Structure

```
Task4/
├── .env                       # Environment variables (API keys, DB URIs, Qdrant credentials)
├── pyproject.toml             # Project metadata and package dependencies
├── continous_agent.py         # Main execution script (RAG setup, SQL tools, agent logic, CLI loop)
├── company-policy.pdf         # Source PDF for general company policies
├── refund-policy.pdf          # Source PDF for refund policies
├── dummy data/                # Sample CSV datasets used to seed PostgreSQL tables
│   ├── customers.csv          # Sample customer records
│   ├── invoices.csv           # Sample customer billing records
│   ├── pending-refunds.csv    # Sample pending refund requests
│   └── refunds.csv            # Sample completed refund entries
└── README.md                  # Project documentation
```

---

## 🗄️ Database & Vector Store Setup

### 1. PostgreSQL Database Tables
The application automatically creates and interacts with the following tables in PostgreSQL:
- **`customers`**: Stores `customer_id`, `name`, `phone_number`, `plan`, `status`, `since`.
- **`invoices`**: Stores `invoice_id`, `customer_id`, `amount`, `invoice_date`, `status`.
- **`pending_refunds`**: Stores refund requests pending human approval (`refund_id`, `customer_id`, `invoice_id`, `amount`, `reason`, `status`, `approver`, `created_at`, `acted_at`).
- **`refunds`**: Stores processed/completed refunds (`refund_id`, `invoice_id`, `amount`, `refund_date`, `status`).
- **LangGraph Checkpoints**: Stores conversation history states (`fieldwork` database).

### 2. Qdrant Vector Store
- **Collection Name**: `task4` (3072-dimensional vector space for `gemini-embedding-2`).
- **PDF Chunking**: `chunk_size=500`, `chunk_overlap=100`.
- Indexed documents are assigned stable deterministic UUIDs (`uuid5`) based on content and metadata to avoid duplicate entries.

---

## 🤖 Agent Tools

| Tool Name | Parameters | Description |
| :--- | :--- | :--- |
| `lookupcustomer` | `phone_number: str` | Fetches customer account ID, name, plan, and status by phone number. |
| `list_invoices` | `customer_id: str` | Retrieves all billing invoices for a given customer. |
| `check_refund_status` | `invoice_id: str` | Checks if a refund exists for a specific invoice ID. |
| `refund_policy` | `query: str` | Queries Qdrant vector index for `refund-policy.pdf` passages. |
| `company_policy` | `query: str` | Queries Qdrant vector index for `company-policy.pdf` passages. |
| `create_refund_request` | `customer_id`, `invoice_id`, `amount`, `reason`, `phone_number` | Creates a refund request. Auto-approves duplicates in same month; otherwise sets to `pending`. |
| `approve_refund` | `refund_id: str`, `approver: str` | Human-in-the-loop tool to approve and finalize a pending refund. |

---

## ⚙️ Environment Variables Setup

Create a `.env` file in the project root with the following configuration:

```env
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Qdrant Vector Database
QDRANT_URL=https://your-qdrant-cluster-url.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here

# LangSmith Observability (Optional)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=Task4
```

---

## 🚀 How to Run

### Prerequisites
1. Ensure Python 3.13+ is installed.
2. PostgreSQL server running locally or remotely with the target databases created (`customer_support` and `fieldwork`).

### 1. Install Dependencies
Using `uv` or `pip`:
```bash
pip install -r pyproject.toml
# or if using uv
uv sync
```

### 2. Run the Agent CLI
Execute the main agent script:
```bash
python continous_agent.py
```

### 3. Interactive CLI Example
```text
Northstar Support Assistant — type 'exit' or 'quit' to stop.

You: Hi, my phone number is +15551234567. Can you check my latest invoice and refund eligibility?
Assistant: Hello! I've looked up your account...
```
