# Python asyncio — Homework Assignment

**Topic:** Concurrent task management beyond `asyncio.gather()`

## Goal

Practise creating, scheduling, monitoring, cancelling, and collecting asynchronous tasks.

You will build a small service-monitoring system that checks multiple servers concurrently.

**Python version:** Python 3.11+ (needed for `asyncio.TaskGroup` and `asyncio.timeout`).

---

## Assignment 1 — Independent task scheduling

**Level:** Beginner

Implement three asynchronous server checks.

### Requirements

1. Each server check takes a different amount of time using `asyncio.sleep()`.
2. Create three tasks using `asyncio.create_task()`.
3. Print a message immediately after scheduling each task.
4. Await the tasks individually.
5. Print their results.

### Starter code

```python
import asyncio


async def check_server(name, delay):
    print(f"Checking {name}...")
    await asyncio.sleep(delay)
    print(f"{name} finished")
    return f"{name}: OK"


async def main():
    # TODO: Create three tasks with asyncio.create_task()

    # TODO: Print a message after scheduling each task

    # TODO: Await each task individually

    # TODO: Print the results


asyncio.run(main())
```

### Questions

- Do the checks run sequentially or concurrently?
- Does creating a task immediately execute the coroutine to completion?
- Does the order in which results are awaited determine the order in which tasks finish?

---

## Assignment 2 — Wait for the first completed task

**Level:** Beginner / Intermediate

Use `asyncio.wait()` with `return_when=asyncio.FIRST_COMPLETED`.

### Requirements

1. Start five server checks with different delays.
2. Wait until at least one task finishes.
3. Print the result of every task in the `done` set.
4. Continue waiting for the remaining tasks.
5. Do not cancel unfinished tasks.

### Questions

- What does `asyncio.wait()` return?
- What is the difference between the `done` and `pending` sets?
- Can more than one task be in `done` after a single wait?

---

## Assignment 3 — Process results as they arrive

**Level:** Intermediate

Use `asyncio.as_completed()`.

### Requirements

1. Start five tasks with different delays.
2. Process each result as soon as it becomes available.
3. Print the completion order.
4. Compare completion order with task creation order.

### Questions

- How does `as_completed()` differ from `gather()`?
- Does it return results in creation order or completion order?
- What happens if one of the tasks raises an exception?

---

## Assignment 4 — Structured concurrency with TaskGroup

**Level:** Intermediate

Rewrite Assignment 2 using `asyncio.TaskGroup`.

### Requirements

1. Start five server checks inside a `TaskGroup`.
2. Make one server check raise an exception.
3. Observe what happens to the other tasks.
4. Handle the resulting `ExceptionGroup` using `except*`.
5. Explain which tasks finish and which are cancelled.

### Questions

- What happens to sibling tasks when one task fails?
- When does the `TaskGroup` context manager exit?
- How does this behavior differ from independently created tasks?

---

## Assignment 5 — Cancellation and timeouts

**Level:** Advanced

Implement a server monitoring system with cancellation and deadlines.

### Requirements

1. Each server check may take up to 10 seconds.
2. Each check has a timeout of 3 seconds.
3. Cancel a task manually when monitoring is stopped.
4. Handle `asyncio.CancelledError` correctly.
5. Use `asyncio.timeout()` to enforce deadlines.
6. Ensure cleanup runs when a task is cancelled or times out.

### Questions

- What is the difference between cancellation and timeout?
- Why should `CancelledError` generally be re-raised after cleanup?
- What happens if a coroutine suppresses cancellation?

---

## Assignment 6 — Build a task scheduler

**Level:** Advanced challenge

Build a scheduler that accepts a list of asynchronous jobs and runs no more than three jobs simultaneously.

### Requirements

1. Accept a list of asynchronous jobs.
2. Run at most three jobs concurrently.
3. Start another job whenever one finishes.
4. Collect successful results.
5. Record failures without stopping other jobs.
6. Support graceful shutdown.
7. Do not use `asyncio.gather()`.

### APIs to investigate

- `asyncio.create_task()`
- `asyncio.wait()`
- `asyncio.Semaphore`
- `asyncio.Queue()`
- `Task.cancel()`

### Questions

- How do you ensure the concurrency limit is never exceeded?
- How can you detect that a job has finished?
- How should the scheduler handle a failed job?
- What does graceful shutdown mean for jobs that are already running?

---

## Suggested order

| Assignment | Main concepts |
|---|---|
| 1 | `create_task()`, `await` |
| 2 | `asyncio.wait()`, `FIRST_COMPLETED` |
| 3 | `asyncio.as_completed()` |
| 4 | `asyncio.TaskGroup`, `ExceptionGroup` |
| 5 | Cancellation, `asyncio.timeout()` |
| 6 | Scheduling, concurrency limits, graceful shutdown |

## Working agreement

Work through the assignments in order. Submit your implementation for review before moving to the next assignment.

The reviewer should prioritize hints and explanations over providing a complete solution. The aim is to understand asyncio's execution model, task lifecycle, cancellation semantics, and structured concurrency—not merely to make the code pass.
