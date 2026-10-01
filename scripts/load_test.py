"""
Zero-dependency load tester using only Python standard library.
No pip install required - works on any Python 3.8+ system.
"""
import urllib.request
import json
import time
import threading

TARGET_URL = "http://localhost:8001/process-batch"
PAYLOAD = json.dumps({
    "reviews": [
        "The battery life on this laptop is absolutely terrible, it barely lasts 2 hours. However, the screen resolution is stunning.",
        "I love this phone! The camera takes amazing pictures in low light, and the battery easily gets me through a full day.",
        "It's okay. The screen is a bit dim outdoors, but it works fine for basic tasks. Nothing special."
    ]
}).encode('utf-8')

HEADERS = {"Content-Type": "application/json"}
stop_flag = threading.Event()
success_count = 0
fail_count = 0
total_requests = 0
lock = threading.Lock()


def worker():
    global success_count, fail_count, total_requests
    req = urllib.request.Request(TARGET_URL, data=PAYLOAD, headers=HEADERS, method='POST')
    while not stop_flag.is_set():
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 200:
                    with lock:
                        success_count += 1
                else:
                    with lock:
                        fail_count += 1
            with lock:
                total_requests += 1
        except Exception:
            with lock:
                fail_count += 1
                total_requests += 1


def run_load_test(num_users=10, duration_seconds=60):
    print(f"Starting load test: {num_users} concurrent users for {duration_seconds} seconds...")
    start_time = time.time()

    threads = []
    for _ in range(num_users):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    time.sleep(duration_seconds)
    stop_flag.set()

    for t in threads:
        t.join()

    total_time = time.time() - start_time
    req_per_sec = total_requests / total_time if total_time > 0 else 0

    print("\n" + "=" * 50)
    print("🚀 LOAD TEST RESULTS")
    print("=" * 50)
    print(f"Total Requests:      {total_requests}")
    print(f"Successful (200 OK): {success_count}")
    print(f"Failed:              {fail_count}")
    print(f"Total Time:          {total_time:.2f} seconds")
    print(f"Throughput:          {req_per_sec:.2f} requests/sec")
    print("=" * 50)


if __name__ == "__main__":
    run_load_test(num_users=10, duration_seconds=60)