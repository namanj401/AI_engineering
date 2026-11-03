import asyncio
import time
import aiohttp


# 1. The Worker Function
async def fetch(session: aiohttp.ClientSession, client_id: int):
    payload = {
        "prompt": f"Write a long story about user {client_id}.",
        "max_tokens": 150,
    }

    async with session.post("http://localhost:8000/generate", json=payload) as response:
        first_token_received = False
        async for line in response.content:
            line_str = line.decode("utf-8").strip()
            if line_str and not first_token_received:
                print(f"[Client {client_id}] First token arrived")
                first_token_received = True

        print(f"[Client {client_id}] Request finished")


# 2. The Concurrency Engine
async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, i) for i in range(50)]
        await asyncio.gather(*tasks)


# 3. Execution & Timing
if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n==========================================")
    print(f"Total time taken to serve all 50 requests: {total_time:.2f} seconds")
    print(f"==========================================")

