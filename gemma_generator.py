# gemma_generator.py
"""
Use Gemma-2B from Ollama as the final generator.

Make sure:
- You have run:  ollama pull gemma:2b
- Ollama server is running (usually automatic on Windows, or `ollama serve`).
"""

import requests
from config import GEMMA_MODEL_NAME


OLLAMA_URL = "http://localhost:11434/api/generate"


def generate_final_answer(question: str, context: str) -> str:
    """
    Call Ollama's gemma:2b model to generate the final answer
    based on question + retrieved context.
    """
    prompt = f"""You are a car recommendation assistant.

You will receive:
- Context: car descriptions, stats (trips, avg_km, fuel_type, maintenance_score), and car-feature relations.
- Question: what the user wants.

Task:
1. Use the context to REASON which car is best for the user's needs.
2. If no car is explicitly marked "best", infer using:
   - higher avg_km for long trips,
   - more trips for reliability,
   - features like "High Mileage", "Suitable for Long Trips", "Comfort".
3. Then give a clear recommendation and explain WHY.

Context:
{context}

User question:
{question}

Now give the best possible answer based on the context and simple reasoning.
"""

    payload = {
        "model": GEMMA_MODEL_NAME,  # "gemma:2b"
        "prompt": prompt,
        "stream": False,
    }

    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()

    # Ollama returns JSON with "response" key
    return data.get("response", "").strip()
