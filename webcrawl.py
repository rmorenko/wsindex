# ruff: noqa: E501, RUF002 - the assignment text below is quoted verbatim
"""
Build an asynchronous web crawler that fetches pages, extracts links, and processes them concurrently. To keep the assignment focused on asyncio, simulate network requests rather than making real HTTP calls.

Scenario
You are implementing a crawler that processes a collection of URLs. Some requests succeed, some fail temporarily, and others fail permanently.

Your crawler must handle these situations efficiently without exceeding a configured concurrency limit.

Requirements
1. Producer–consumer architecture
   - Use asyncio.Queue to distribute URLs to workers.
   - Start 3 concurrent worker coroutines.
   - Workers continuously retrieve URLs from the queue.
2. Simulated HTTP requests
   - Each request takes a random 0.5–2.0 seconds.
   - Each request has a 20% probability of a temporary failure.
   - Each request has a 10% probability of a permanent failure.
   - Otherwise, the request succeeds and returns a simulated page title.
3. Retries
   - Retry temporary failures up to 3 attempts in total.
   - Wait before retrying using exponential backoff: 0.5, 1.0 seconds.
   - Never retry permanent failures.
   - A failed request must not terminate the other workers.
4. Concurrency control
   - Use asyncio.Semaphore to ensure no more than 2 HTTP requests are active simultaneously.
   - Retries must also respect this limit.
   - Do not hold the semaphore while waiting for retry backoff.
5. Timeouts
   - Each individual HTTP attempt must have a timeout of 1.5 seconds.
   - Treat a timeout as a temporary failure eligible for retry.
6. Results and statistics
   - Collect successful results and failed URLs.
   - Report the number of successful URLs, permanently failed URLs, and URLs that exhausted their retries.
   - Report total execution time.

"""

import asyncio
import random
import time
import uuid
from asyncio.queues import QueueShutDown

URLS = [f"https://example.com/page/{i}" for i in range(1, 100)]

WORKERS = 3
MAX_CONCURRENT_REQUESTS = 2
MAX_ATTEMPTS = 3
REQUEST_TIMEOUT = 1.5


class TemporaryError(Exception):
    pass


class PermanentError(Exception):
    pass


class RetriesExhaustedError(Exception):
    pass


async def fetch_page(url):
    delay = random.uniform(0.5, 2.0)
    await asyncio.sleep(delay)

    outcome = random.random()

    if outcome < 0.2:
        raise TemporaryError(url)

    if outcome < 0.3:
        raise PermanentError(url)

    return {"url": url, "title": f"Title of {url}"}


async def worker(queue, semaphore, results):
    worker_name = f"Worker_{uuid.uuid4()}"
    try:
        while True:
            url = await queue.get()
            print(f"Worker {worker_name} working on {url}")
            results.append(await process_url(url, semaphore))
    except QueueShutDown:
        pass


async def process_url(url, semaphore):
    backoff = 0.5
    retries = MAX_ATTEMPTS
    while True:
        try:
            async with semaphore:
                return await asyncio.wait_for(fetch_page(url), REQUEST_TIMEOUT)
        except (TimeoutError, TemporaryError):
            retries -= 1
            if retries == 0:
                return {"url": url, "error": RetriesExhaustedError()}
            await asyncio.sleep(backoff)
            backoff *= 2
        except PermanentError as e:
            return {"url": url, "error": e}


def create_report(results) -> dict[str, float]:
    exhausted = 0
    successful = 0
    permanent = 0
    for result in results:
        if not result.get("error"):
            successful += 1
        elif isinstance(result["error"], RetriesExhaustedError):
            exhausted += 1
        else:
            permanent += 1
    return {"exhausted": exhausted, "successful": successful, "permanent": permanent}


async def main():
    queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    results = []

    start = time.perf_counter()
    for url in URLS:
        await queue.put(url)
    queue.shutdown()
    await asyncio.gather(*[worker(queue, semaphore, results) for _ in range(WORKERS)])
    elapsed = time.perf_counter() - start
    report = create_report(results)
    report["elapsed"] = elapsed
    print(report)


if __name__ == "__main__":
    random.seed(444)
    asyncio.run(main())
