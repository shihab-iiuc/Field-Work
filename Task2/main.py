@
from dotenv import load_dotenv
load_dotenv(override=True)
# Initialize Models
from langchain.chat_models import init_chat_model

# Primary Model (Gemini)
gemini = init_chat_model(
    "google_genai:gemini-3.1-flash-lite"
)
# Fallback Model (Groq)
groq = init_chat_model(
    "groq:openai/gpt-oss-120b"
)
model = gemini.with_fallbacks([groq])


#System Prompt Loading
import yaml
from pathlib import Path

PROMPTS = yaml.safe_load(
    Path("prompt.yml").read_text(encoding="utf-8")
)

SYSTEM = PROMPTS["RESEARCH_SYSTEM"]


#TOOLS & Agent  creation
from langchain_core.messages import HumanMessage

def make_bullet_points(text: str) -> str:
    """
    Convert text into clean bullet points.
    """

    bullets = [
        line.strip("-• ")
        for line in text.split("\n")
        if line.strip()
    ]

    return "\n".join(f"• {b}" for b in bullets)


def describe_image(image_url: str) -> str:
    """
    Describe an image from a public URL.

    Use this tool when the user provides:
    - image URL
    - research figure
    - graph
    - chart
    - diagram
    - screenshot
    """

    response = model.invoke([
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Describe this image in beginner-friendly language. "
                        "If it is a research figure, explain the important "
                        "visual elements and summarize the key findings."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": image_url,
                },
            ]
        )
    ])

    return response.content



from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[
        make_bullet_points,
        describe_image
    ],
    system_prompt=SYSTEM,
)


#Userside
query = """
Explain Reinforcement Learning with bullet points,
Describe the image at this URL: https://www.bing.com/images/search?view=detailV2&ccid=ad2ORulq&id=CF476BC1844E6A84DE10675C39F7EFD6BAF2DBF1&thid=OIP.ad2ORulqXN7eNStxcZyU1wHaHa&mediaurl=https%3A%2F%2Fimg.freepik.com%2Ffree-vector%2Fmachine-learning-isometric-composition-with-text-cubes-with-code-robotic-head-chip-with-academic-hat-vector-illustration_1284-82015.jpg%3Fsize%3D626%26ext%3Djpg&cdnurl=https%3A%2F%2Fth.bing.com%2Fth%2Fid%2FR.69dd8e46e96a5cdede352b71719c94d7%3Frik%3D8dvyutbv9zlcZw%26pid%3DImgRaw%26r%3D0&exph=626&expw=626&q=photos+for+machine+learning+with+jpg+extension+link&form=IRPRST&ck=18F3B716ED6E1E1F0B8A4382D4E6DF70&selectedindex=3&itb=0&cw=1145&ch=499&ajaxhist=0&ajaxserp=0&vt=0&sim=11,

"""

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ]
    }
)

print("\n========== Final Answer ==========\n")
print(result["messages"][-1].text)