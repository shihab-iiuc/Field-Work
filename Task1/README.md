# AI Research Assistant

A lightweight AI-powered Research Assistant built with **Google Gemini 3.1 Flash Lite**, **Pydantic Structured Output**, and **Python**. It generates well-structured research responses in a consistent format, making research easier for students, researchers, and developers.

---

## Features

- Explain research concepts in simple language
- Summarize research papers
- Explain research methods
- Analyze research gaps
- Generate literature overviews
- Answer general research questions
- Structured JSON output using Pydantic
- Token usage and estimated cost tracking
- Easy to customize with your own system prompt

---

## Output Format

The assistant returns structured responses containing:

- **Task Type**
- **Topic**
- **Goal**
- **Key Points**
- **Examples**
- **Follow-up Suggestions**

---

## Project Structure

```
.
├── RA.ipynb              # Main notebook
├── prompt.yml            # System prompt
├── .env                  # API key
└── README.md
```

---

## Requirements

- Python 3.10+
- Google Gemini API Key

Install dependencies:

```bash
pip install google-genai python-dotenv pydantic pyyaml
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### 2. Create a `.env` file

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

### 3. Configure the system prompt

Edit `prompt.yml` to customize the assistant's behavior.

---

## How to Use

Run the notebook or call the assistant directly.

Example:

```python
result = research_assistant(
    "Explain overfitting in machine learning with simple examples."
)

draft_reply(result)
```

---

## Supported Tasks

The assistant automatically identifies the request type and supports:

- Concept Explanation
- Paper Summary
- Method Explanation
- Research Gap Analysis
- Literature Overview
- General Research Help

---

## Who Can Use It?

This project is suitable for:

- Students
- Researchers
- Thesis writers
- University teachers
- AI/ML learners
- Software developers
- Anyone seeking structured research assistance

---

## Cost Tracking

The project tracks:

- Input tokens
- Output tokens
- Estimated API cost

---

## Technologies Used

- Python
- Google Gemini 3.1 Flash Lite
- Google GenAI SDK
- Pydantic
- PyYAML
- python-dotenv

---



---

## License

This project is available under the MIT License.
