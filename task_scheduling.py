import asyncio
import random


async def check_server(name, delay):
    print(f"Checking {name}...")
    await asyncio.sleep(delay)
    print(f"{name} finished")
    return f"{name}: OK"


async def check_server_with_timeout(name, delay):
    print(f"Checking {name}...")
    try:
        async with asyncio.timeout(3):
            await asyncio.sleep(delay)
            print(f"{name} finished")
            return f"{name}: OK"
    except TimeoutError:
        print(f"{name} TIMEOUT")
        return f"{name}: TIMEOUT"
    except asyncio.CancelledError:
        print(f"{name} CANCELED")
        raise


async def check_server(name, delay):
    print(f"Checking {name}...")
    await asyncio.sleep(delay)
    print(f"{name} finished")
    return f"{name}: OK"


async def check_server_with_error(name, delay):
    print(f"Checking {name}...")
    await asyncio.sleep(delay)
    raise Exception(name)


async def main():
    print("Assignment 1")
    # Create three tasks with asyncio.create_task()
    tasks = []
    for i in range(3):
        # Print a message after scheduling each task
        tasks.append(asyncio.create_task(check_server(f"Task {i}", 2)))
        print(f"Task {i} scheduled")

    # Await each task individually
    for task in tasks:
        await task
    # Print the results
    for task in tasks:
        print(task.result())

    print("Assignment 2")
    # Start five server checks with different delays.
    tasks = create_tasks()
    # Wait until at least one task finishes.
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    # Print the result of every task in the `done` set.
    print_results(done)
    # Continue waiting for the remaining tasks.
    done, _ = await asyncio.wait(pending)
    print_results(done)
    print("Assignment 3")
    # Start five tasks with different delays.
    tasks = create_tasks()
    # 3. Print the completion order.
    # 4. Compare completion order with task creation order.
    async for task in asyncio.as_completed(tasks):
        print(task.result())
    print("Assignment 4")
    # 3. Observe what happens to the other tasks.
    # 4. Handle the resulting `ExceptionGroup` using `except*`.
    # 5. Explain which tasks finish and which are cancelled.
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(check_server(f"Task {i}", random.randint(0, i) + 1))
                for i in range(4)
            ]
            tasks.append(group.create_task(check_server_with_error("Task 4", 2)))
    except ExceptionGroup as exc:
        print(exc.exceptions)
    for task in tasks:
        print(task)
    print("Assignment 5")
    tasks = []
    for i in range(10):
        tasks.append(
            asyncio.create_task(check_server_with_timeout(f"Task {i}", random.randint(1, 10)))
        )
        await asyncio.sleep(0.5)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    print("Assignment 6")


def create_tasks(number=4):
    return [
        asyncio.create_task(check_server(f"Task {i}", random.randint(0, i) + 1))
        for i in range(number)
    ]


def print_results(done):
    for result in done:
        print(result.result())


random.seed(666)
asyncio.run(main())
