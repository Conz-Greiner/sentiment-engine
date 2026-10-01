"""
High-throughput async processor for cloud GPU environments (24GB+ VRAM).
Processes 10,000 real Amazon reviews with 64 concurrent workers.
"""
import asyncio
import httpx
import pandas as pd
import time
import json
import math
from datasets import load_dataset
from collections import Counter

# Configuration for MAXIMUM throughput on 24GB VRAM
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "granite-3.2-2b"
TOTAL_REVIEWS = 10000
MAX_CONCURRENT_REQUESTS = 64  # 64 concurrent single-review requests

# Relaxed JSON Schema to prevent tokenization conflicts with strict enums
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "description": "positive, negative, or neutral"},
        "mentioned_features": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"}
    },
    "required": ["sentiment", "mentioned_features", "summary"]
}

SYSTEM_PROMPT = """You are an expert e-commerce data analyst. Analyze the review and output ONLY valid JSON matching this schema.
Use lowercase for sentiment: "positive", "negative", or "neutral".
Do not include markdown formatting. Output ONLY the raw JSON object."""


async def process_review(client: httpx.AsyncClient, review: str, review_idx: int, semaphore: asyncio.Semaphore):
    async with semaphore:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Review: {review}"}
            ],
            "temperature": 0.0,
            "max_tokens": 150,
            "response_format": {"type": "json_object"},
            "extra_body": {"guided_json": JSON_SCHEMA}
        }

        try:
            response = await client.post(API_URL, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip().replace("```json", "").replace("```", "")

            parsed = json.loads(content)
            return {"review_idx": review_idx, "status": "success", "review": review, "data": parsed}
        except Exception as e:
            return {"review_idx": review_idx, "status": "failed", "review": review, "error": str(e)}


async def main():
    print(f"📥 Loading {TOTAL_REVIEWS} real Amazon reviews from Hugging Face...")
    dataset = load_dataset("fancyzhx/amazon_polarity", split=f"train[:{TOTAL_REVIEWS}]")
    reviews = dataset["content"]

    print(f"📦 Preparing {len(reviews)} individual review tasks.")
    print(f"🚀 Starting high-throughput processing ({MAX_CONCURRENT_REQUESTS} concurrent workers)...\n")

    results = []
    start_time = time.time()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with httpx.AsyncClient() as client:
        tasks = [process_review(client, review, i, semaphore) for i, review in enumerate(reviews)]

        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            success_count = sum(1 for r in results if r["status"] == "success")
            print(f"\rProgress: {len(results)}/{len(reviews)} reviews | Success: {success_count}", end="")

    total_time = time.time() - start_time
    successful_reviews = [r for r in results if r["status"] == "success"]

    reviews_processed = len(successful_reviews)
    req_per_sec = reviews_processed / total_time if total_time > 0 else 0
    estimated_tokens_per_sec = req_per_sec * 100

    print("\n\n" + "=" * 70)
    print("🚀 CLOUD HIGH-THROUGHPUT RESULTS (10,000 Reviews)")
    print("=" * 70)
    print(f"Model:                   IBM Granite 3.2 2B Instruct")
    print(f"Total Reviews Processed: {reviews_processed}")
    print(f"Successful Requests:     {reviews_processed}/{len(reviews)}")
    print(f"Total Time:              {total_time:.2f} seconds ({total_time/60:.2f} mins)")
    print(f"Review Throughput:       {req_per_sec:.2f} reviews/sec")
    print(f"Estimated Token Speed:   ~{estimated_tokens_per_sec:.0f} tokens/sec")
    print("=" * 70)

    # Save to CSV with robust NaN handling
    df_data = []
    for r in successful_reviews:
        sentiment = r["data"].get("sentiment", "unknown")
        # Normalize sentiment to lowercase and handle NaN
        if isinstance(sentiment, float) and math.isnan(sentiment):
            sentiment = "unknown"
        else:
            sentiment = str(sentiment).strip().lower()
            # Map common variations to standard labels
            if "pos" in sentiment:
                sentiment = "positive"
            elif "neg" in sentiment:
                sentiment = "negative"
            elif sentiment in ["neutral", "mixed", "neutral-negative"]:
                sentiment = "neutral"

        df_data.append({
            "review": r["review"],
            "sentiment": sentiment,
            "features": str(r["data"].get("mentioned_features", [])),
            "summary": str(r["data"].get("summary", ""))
        })

    df = pd.DataFrame(df_data)
    df.to_csv("cloud_results.csv", index=False)
    print(f"\n✅ Results saved to cloud_results.csv")

    sentiments = Counter(df["sentiment"])
    print("\n📈 Sentiment Distribution:")
    for sentiment, count in sentiments.most_common():
        print(f"  {str(sentiment):<12}: {count:5d} ({count/len(df)*100:5.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())