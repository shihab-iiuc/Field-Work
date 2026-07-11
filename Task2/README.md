# AI Research Assistant

An AI Research Assistant built with **LangChain** that uses **Google Gemini** as the primary model and **Groq GPT-OSS-120B** as a fallback. The assistant can answer research questions, generate bullet-point summaries, and describe images from public URLs.

## Features

* Google Gemini as the primary LLM
* Automatic fallback to Groq GPT-OSS-120B
* LangChain agent with tool calling
* YAML-based system prompt
* Environment variable support with `.env`
* Bullet-point generation
* Image description from public URLs
* Modular and extensible design

## Available Tools

### `make_bullet_points()`

Converts text into clean bullet points.

### `describe_image()`

Analyzes and describes an image from a public URL in simple language.

## Project Structure

```text
.
├── app.py
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

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
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

## Workflow

```text
User Query
    │
    ▼
LangChain Agent
    │
    ├── Tool Required → Execute Tool
    │
    └── No Tool → Generate Response
              │
              ▼
      Gemini (Primary)
              │
          On Failure
              ▼
     Groq GPT-OSS-120B
              │
              ▼
       Final Response
```

## Future Improvements

* PDF support
* Web search
* RAG integration
* OCR support
* Conversation memory
* Streaming responses

## License

MIT License.
