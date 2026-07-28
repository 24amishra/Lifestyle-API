"""
test.py – Quick smoke test for the health-coach LLM call.

Run standalone:  python test.py
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Example Fitbit-style health data for testing
SAMPLE_HEALTH_DATA = {
    "activity": {
        "steps": 22000,
        "distance_km": 7.8,
        "floors": 12,
        "active_minutes": 65,
        "calories_out": 2320,
    },
    "weight": {"weight_kg": 62.3, "bmi": 22.4, "date": "2025-05-20"},
    "heartrate": {
        "resting_heart_rate": 61,
        "average_heart_rate": 74,
        "max_heart_rate": 142,
    },
    "cardio_fitness": {"vo2_max": 38.2, "fitness_score": "Good"},
}

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=openai_key,
)

messages = [
    (
        "system",
        "You are a helpful assistant in charge of helping a cancer patient. "
        "Either raise or lower their step estimation based on this data.",
    ),
    (
        "human",
        f"""Here is the health data:

{SAMPLE_HEALTH_DATA}

Their step goal is 20000. Should we raise or lower the estimation? Provide a new threshold.""",
    ),
]

if __name__ == "__main__":
    ai_msg = llm.invoke(messages)
    print(ai_msg.content)
