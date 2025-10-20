import requests
import time
import numpy as np
import random
import sys

# --- Configuration ---
# The URL of your running FastAPI service
API_URL = "http://localhost:8000/classify"
# Number of requests to send for the benchmark
NUM_REQUESTS = 200
# Number of warm-up requests to discard (helps stabilize results)
WARMUP_REQUESTS = 10

# A diverse set of sample texts to avoid caching effects
SAMPLE_TEXTS = [
    "This is a perfectly safe and normal message.",
    "I'm so excited about the new project launch next week!",
    "The weather today is just beautiful, isn't it?",
    "Can you please review the document I sent over this morning?",
    "Let's schedule a meeting to discuss the quarterly results.",
    "این یک پیام کاملا عادی و بدون مشکل است",
    "خیلی برای این پروژه جدید هیجان زده هستم",
    "امیدوارم روز خوبی داشته باشید",
    "لطفا گزارش مالی را بررسی کنید",
    "You are such an idiot, I can't believe you said that.",
    "This is the worst service I have ever experienced.",
    "I absolutely despise this kind of behavior from people.",
    "Get out of my country, you don't belong here.",
    "چه آدم احمقی هستی، این حرفا چیه میزنی؟",
    "از این شرکت و خدماتش متنفرم",
    "این گروه از آدما خیلی چندش آورن",
]


def run_benchmark():
    """
    Sends a series of requests to the API and records their latencies.
    """
    latencies = []
    print("--- Starting Latency Benchmark ---")

    # 1. Check if the server is running
    try:
        response = requests.get("http://localhost:8000/health")
        if (
            response.status_code != 200
            or not response.json().get("status") == "healthy"
        ):
            print("Error: API is not healthy. Please start the server.")
            sys.exit(1)
        print("✅ API is running and healthy.")
    except requests.exceptions.ConnectionError:
        print(
            "\n❌ Error: Could not connect to the API server at http://localhost:8000."
        )
        print(
            "Please ensure the FastAPI application is running before executing this script."
        )
        print("You can run it with: uvicorn app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    # 2. Warm-up phase
    print(
        f"\n🚀 Sending {WARMUP_REQUESTS} warm-up requests (these will not be timed)..."
    )
    for _ in range(WARMUP_REQUESTS):
        text_to_send = random.choice(SAMPLE_TEXTS)
        requests.post(API_URL, json={"text": text_to_send})
    print("Warm-up complete.")

    # 3. Main benchmark loop
    print(f"\n📊 Running main benchmark with {NUM_REQUESTS} requests...")
    for i in range(NUM_REQUESTS):
        # Select a random text to send
        text_to_send = random.choice(SAMPLE_TEXTS)
        payload = {"text": text_to_send}

        try:
            # Record time right before and after the request
            start_time = time.perf_counter()
            response = requests.post(API_URL, json=payload)
            end_time = time.perf_counter()

            if response.status_code == 200:
                # Convert duration to milliseconds and store it
                duration_ms = (end_time - start_time) * 1000
                latencies.append(duration_ms)
                # Print progress indicator
                print(".", end="", flush=True)
                if (i + 1) % 50 == 0:
                    print(f" [{i+1}/{NUM_REQUESTS}]")  # Newline every 50 requests
            else:
                print(
                    f"\nError on request {i+1}: Status {response.status_code} - {response.text}"
                )
        except Exception as e:
            print(f"\nAn exception occurred during request {i+1}: {e}")

    print("\n\nBenchmark finished. Calculating results...")
    return latencies


def analyze_and_report(latencies: list):
    """
    Calculates key performance metrics from the list of latencies and prints a report.
    """
    if not latencies:
        print("\nNo successful requests were recorded. Cannot generate a report.")
        return

    # Calculate metrics using NumPy
    p50 = np.percentile(latencies, 50)  # Median
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    avg = np.mean(latencies)
    std_dev = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    # Calculate Requests per Second (RPS) as a measure of throughput
    total_time_seconds = sum(latencies) / 1000
    rps = len(latencies) / total_time_seconds if total_time_seconds > 0 else 0

    # Print the final report
    print("\n--- 📈 Latency Benchmark Report ---")
    print(f"Total Successful Requests: {len(latencies)} out of {NUM_REQUESTS}")
    print("-" * 35)
    print(f"Average Latency: {avg:.2f} ms")
    print(f"Median Latency (p50): {p50:.2f} ms")
    print(f"95th Percentile (p95): {p95:.2f} ms")
    print(f"99th Percentile (p99): {p99:.2f} ms")
    print(f"Min Latency: {min_latency:.2f} ms")
    print(f"Max Latency: {max_latency:.2f} ms")
    print(f"Standard Deviation: {std_dev:.2f} ms")
    print("-" * 35)
    print(f"Requests Per Second (RPS): {rps:.2f}")
    print("-----------------------------------")


if __name__ == "__main__":
    # Ensure this script is run on a CPU-only environment for accurate reporting
    # You can force this by setting an environment variable before running the API
    # Example: CUDA_VISIBLE_DEVICES="" uvicorn app:app --host 0.0.0.0 --port 8000

    recorded_latencies = run_benchmark()
    analyze_and_report(recorded_latencies)
