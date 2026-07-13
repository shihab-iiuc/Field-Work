# AI Research Assistant

A lightweight AI Research Assistant built with **LangChain**, using **Google Gemini** as the primary model and **Groq GPT-OSS-120B** as an automatic fallback. The assistant answers research questions, summarizes text into bullet points, describes images from public URLs, and uses **LangSmith** for tracing agent and tool execution.

## Features

* Google Gemini as the primary LLM
* Automatic fallback to Groq GPT-OSS-120B
* LangChain Agent with tool calling
* LangSmith tracing for agents and tools
* YAML-based system prompt
* Environment variable support (`.env`)
* Bullet-point generation
* Image description from public URLs
* Modular and extensible architecture

## Tech Stack

* Python
* LangChain
* LangSmith
* Google Gemini
* Groq
* PyYAML
* python-dotenv

## Project Structure

```text
.
├── venv
├── main.py
├── prompt.yml
├── .env
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key

LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=AI-Research-Assistant
```

## Configuration

Configure the system prompt in `prompt.yml`.

```yaml
RESEARCH_SYSTEM: |
  You are an AI Research Assistant.
  Answer clearly and accurately.
```

## Usage

Run the application:

```bash
python app.py
```

Example query:

```python
query = """
Explain Reinforcement Learning with bullet points.

Describe the image at this URL:
https://example.com/image.jpg
"""
```

## Available Tools

| Tool                   | Description                                        |
| ---------------------- | -------------------------------------------------- |
| `make_bullet_points()` | Converts text into concise bullet points.          |
| `describe_image()`     | Describes and explains an image from a public URL. |

## Workflow

```text
User Query
      │
      ▼
LangChain Agent
      │
      ├── Tool Selection
      │
      ├── Execute Tool(s)
      │
      ▼
Gemini (Primary)
      │
      ├── Success → Response
      │
      └── Failure
             │
             ▼
     Groq GPT-OSS-120B
             │
             ▼
      Final Response
             │
             ▼
      LangSmith Tracing
```

## LangSmith Tracing

LangSmith records the complete execution of the application, including:

* Agent execution
* Tool calls
* LLM requests and responses
* Execution timeline
* Token usage
* Errors and debugging information

This makes it easier to monitor, debug, and evaluate the agent's behavior.


