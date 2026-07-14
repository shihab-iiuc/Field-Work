# Northstar Support Assistant

A terminal-based AI customer support agent for **Northstar Services**, built with LangChain / LangGraph and Google Gemini. It looks up customers, lists invoices, checks refund status, and can create or approve refund requests — all backed by a PostgreSQL database, with conversation memory persisted across turns via `PostgresSaver`.

## What it does

The assistant is a LangChain agent (`create_agent`) with access to 6 tools:

| Tool | Purpose |
|---|---|
| `lookupcustomer` | Find a customer by phone number (id, name, plan, status, signup date) |
| `list_invoices` | List all invoices for a customer_id |
| `check_refund_status` | Check if a refund exists for an invoice, and its status |
| `refund_policy` | Return the hardcoded refund policy text |
| `create_refund_request` | Create a refund request; auto-approves if the customer was charged 2+ times in the same month, otherwise leaves it pending |
| `approve_refund` | Human approves a pending refund, which executes it |

Conversation state is checkpointed to Postgres (`langgraph.checkpoint.postgres.PostgresSaver`), so the agent remembers the ongoing conversation for a given `thread_id` between messages — you can chat with it continuously in the terminal, similar to a chat app.

## Project structure

```
.
├── main.py           # entry point — agent, tools, DB setup, chat loop
├── requirements.txt  # Python dependencies
├── .env              # your API key (not committed — see below)
└── README.md
```

Everything currently lives in `main.py`: model setup, database connection, tool definitions, agent creation, and the interactive chat loop.

## Prerequisites

- Python 3.10+
- PostgreSQL running locally (or reachable) with a `customer_support` database containing `customers` and `invoices` tables
- A Gemini API key

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv venv

# Activate it
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, create one with:

```
python-dotenv
langchain
langchain-google-genai
langgraph
langgraph-checkpoint-postgres
sqlalchemy
psycopg2-binary
```

### 3. Set up your `.env` file

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 4. Connect the database

The project uses **two** Postgres connections:

- `customer_support` database — holds `customers`, `invoices`, and the `pending_refunds` / `refunds` tables (the last two are auto-created on first run if they don't exist).
- `fieldwork` database — used purely as the checkpoint store for conversation memory (`PostgresSaver`).

Both are configured directly in `main.py`. Update these connection strings to match your local setup:

```python
# main.py — customer/invoice data
DATABASE_URL = "postgresql+psycopg2://postgres:1234@localhost:5432/customer_support"

# main.py — conversation checkpoint storage
DB_URI = "postgresql://postgres:1234@localhost:5432/fieldwork?sslmode=disable"
```

Make sure both databases exist before running:

```sql
CREATE DATABASE customer_support;
CREATE DATABASE fieldwork;
```

The `customer_support` database needs `customers` and `invoices` tables populated with your own data (the agent expects columns like `phone_number`, `customer_id`, `name`, `plan`, `status`, `since` on `customers`, and `invoice_id`, `customer_id`, `amount`, `invoice_date`, `status` on `invoices`). `pending_refunds` and `refunds` are created automatically by the script.

## Usage

With your virtual environment activated and both databases set up:

```bash
python main.py
```

You'll see:

```
Northstar Support Assistant — type 'exit' or 'quit' to stop.

You: 
```

Type your message and press Enter. The agent will respond, and the conversation continues in the same thread (`thread_id="customer-session-2"` by default) so it remembers prior context. Type `exit` or `quit` to end the session.

Example:

```
You: Hi, my phone number is +8801xxxxxxxxx, do I have any pending refunds?

Assistant

Let me look that up for you...
```

## Notes

- The refund flow is human-in-the-loop by default: `create_refund_request` only auto-approves when a duplicate-monthly-charge rule is met; otherwise it stays `pending` until someone calls `approve_refund`.
- Credentials and connection strings in this README/`main.py` are placeholders — replace them with your own before deploying, and avoid committing real credentials to git (use `.env` / environment variables instead).
