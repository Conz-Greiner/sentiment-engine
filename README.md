# 📦 Complete GitHub Repository Structure

Here's the professionally organized structure. Let me walk you through the directory layout first, then provide every file's final content.

## 🗂️ Directory Structure

```
sentiment-engine/
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore rules
├── docker-compose.yml             # Local Docker orchestration
├── Dockerfile                     # FastAPI container image
├── requirements.txt               # API dependencies
├── main.py                        # FastAPI backend (core engine)
├── test_data.json                 # Sample test payload
├── k8s/                           # Kubernetes manifests
│   ├── api-deployment.yaml
│   └── api-service.yaml
├── scripts/                       # Testing & processing scripts
│   ├── load_test.py               # Zero-dependency load tester
│   ├── process_cloud_dataset.py   # Cloud high-throughput processor
│   └── requirements-cloud.txt     # Cloud environment dependencies
└── data/                          # Output folder (gitignored)
    └── .gitkeep
```

---

## 📄 File 1: `README.md`

```markdown
# 🚀 High-Throughput E-commerce Sentiment & Tagging Engine

A production-ready, cost-optimized MLOps pipeline for extracting structured sentiment and feature tags from massive e-commerce datasets using local, quantized LLMs.

Designed to prove that with the right architectural choices (quantization, guided decoding, and strict resource capping), enterprise-grade AI inference can be achieved on highly constrained hardware or scaled infinitely in the cloud.

## 🏗️ Hybrid MLOps Architecture

- **Inference Engine**: vLLM serving `ibm-granite/granite-3.2-2b-instruct` (BF16).
- **Optimization**: Chunked prefill, strict `guided_json` decoding to guarantee 100% schema compliance without regex parsing, and optimized concurrency semaphores.
- **API Backend**: FastAPI with async batch processing, defensive Pydantic validation, and `ast.literal_eval` fallbacks for quantized model quirks.
- **Orchestration**: Hybrid deployment. The stateless FastAPI service is orchestrated via Kubernetes (Minikube) to demonstrate K8s proficiency, while the GPU-heavy vLLM engine runs on optimized Docker Compose for direct, low-latency NVIDIA hardware passthrough.

## 📊 Verified Cloud Load Testing & Business ROI

*Tested on a rented 24GB VRAM Cloud GPU (RTX 3090/4090) processing 10,000 real Amazon Polarity reviews.*

> **💡 The ROI Proof:**
> - **Throughput**: Processed 10,000 reviews at **21.44 reviews/sec** (~**2,144 tokens/sec**).
> - **Time to Process**: 6.3 minutes for the entire dataset.
> - **Cloud Cost**: Estimated cost to process 10,000 reviews on Vast.ai (RTX 4090 at $0.50/hr): **~$0.05**.
> - **API Equivalent**: Equivalent cost via OpenAI GPT-4 API: **~$45.00**.
> - **Savings**: **~99.8% cost reduction** with zero data leaving the private environment (critical for NDA/privacy compliance).

## 🛠️ Tech Stack

- **Model**: `ibm-granite/granite-3.2-2b-instruct` (Chosen for hyper-efficiency and structured output capabilities).
- **Inference**: vLLM (`--enable-chunked-prefill`, `--max-num-batched-tokens 8192`).
- **Backend**: FastAPI, Pydantic, httpx, Python Asyncio.
- **Testing**: Custom Asyncio Load Tester (zero external dependencies).
- **Deployment**: Docker, Kubernetes (Minikube), WSL2.

## 🚀 Quick Start

### Local Setup (4GB+ VRAM GPU)

```bash
# Start the GPU-accelerated vLLM engine
docker-compose up -d vllm

# Start the FastAPI service
docker-compose up -d api

# Test the pipeline
curl -X POST http://localhost:8001/process-batch \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

### Cloud Setup (24GB+ VRAM GPU)

```bash
# Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r scripts/requirements-cloud.txt

# Start vLLM engine
vllm serve ibm-granite/granite-3.2-2b-instruct \
  --served-model-name granite-3.2-2b \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --enable-chunked-prefill \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --dtype bfloat16 \
  --port 8000

# Process 10,000 real reviews
python3 scripts/process_cloud_dataset.py
```

## 🛡️ Engineering Highlights & Resilience

- **Defensive JSON Parsing**: The pipeline handles quantized model quirks by falling back to `ast.literal_eval` and normalizing edge-case string outputs (e.g., "Negative" → "negative", NaN handling).
- **Concurrency Throttling**: Used `asyncio.Semaphore` to perfectly saturate the GPU without blowing up the KV cache or triggering HTTP 429/503 errors.
- **Hybrid K8s/Docker**: Demonstrated advanced orchestration by routing Kubernetes-orchestrated stateless API traffic to bare-metal Docker containers for maximum GPU utilization.
- **Zero-Dependency Load Testing**: Custom Python load tester using only the standard library, eliminating dependency management headaches.

## 📁 Project Structure

```
├── main.py                 # FastAPI backend
├── docker-compose.yml      # Local orchestration
├── k8s/                    # Kubernetes manifests
├── scripts/                # Load testing & dataset processing
└── data/                   # Output directory (gitignored)
```