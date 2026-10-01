import os
import json
import ast
import httpx
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(title="E-commerce Sentiment Engine")


class ReviewAnalysis(BaseModel):
    sentiment: str = Field(description="Must be exactly 'positive', 'negative', or 'neutral'")
    mentioned_features: List[str] = Field(description="List of specific product features mentioned")
    summary: str = Field(description="A concise 1-sentence summary of the review")


class BatchRequest(BaseModel):
    reviews: List[str]


VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "granite-2b")

# vLLM expects the schema as a dict or string for guided decoding
GUIDED_JSON_SCHEMA = ReviewAnalysis.model_json_schema()

SYSTEM_PROMPT = """You are an expert e-commerce data analyst. Analyze the provided customer review.
You MUST output ONLY a valid JSON object with the following exact keys (all lowercase):
{
  "sentiment": "positive" or "negative" or "neutral",
  "mentioned_features": ["feature1", "feature2"],
  "summary": "A concise 1-sentence summary."
}
Do not include markdown formatting (like ```json). Output ONLY the raw JSON object."""


@app.post("/process-batch")
async def process_batch(request: BatchRequest):
    results = []

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ""}
        ],
        "temperature": 0.0,  # Lower temperature for strict, deterministic JSON
        "max_tokens": 150,
        "response_format": {"type": "json_object"},  # OpenAI standard JSON mode
        "extra_body": {
            "guided_json": GUIDED_JSON_SCHEMA  # vLLM specific strict token-level enforcement
        }
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, review in enumerate(request.reviews):
            payload["messages"][1]["content"] = f"Review: {review}"

            try:
                response = await client.post(VLLM_URL, json=payload)
                response.raise_for_status()
                data = response.json()

                content = data["choices"][0]["message"]["content"].strip()

                # Clean up markdown if the model slips
                content = content.replace("```json", "").replace("```", "").strip()

                # DEFENSIVE PARSING: Try standard JSON first, fallback to Python dict eval
                try:
                    parsed_dict = json.loads(content)
                except json.JSONDecodeError:
                    try:
                        parsed_dict = ast.literal_eval(content)
                    except Exception:
                        raise ValueError(f"Failed to parse content as JSON or dict: {content}")

                # Validate against Pydantic (ensures types and required fields are correct)
                parsed_data = ReviewAnalysis.model_validate(parsed_dict)

                results.append({
                    "review_index": i,
                    "original_review": review,
                    **parsed_data.model_dump()
                })
            except Exception as e:
                results.append({
                    "review_index": i,
                    "original_review": review,
                    "error": str(e)
                })

    df = pd.DataFrame(results)
    csv_path = "/app/data/output_results.csv"

    # Append to CSV, create header if file doesn't exist
    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

    return {"status": "success", "processed_count": len(results), "saved_to": csv_path}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model": MODEL_NAME}