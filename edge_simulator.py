import asyncio
import aiohttp
import random
import datetime

# Configuration
NUM_GATE_NODES = 10
NUM_FOOD_STALL_NODES = 20
NUM_BATHROOM_NODES = 20
TARGET_URL = "http://localhost:8000/api/telemetry"

async def run_node(node_id: str, node_type: str, session: aiohttp.ClientSession):
    while True:
        # Determine delay before next telemetry (2 to 5 seconds)
        delay = random.uniform(2.0, 5.0)
        await asyncio.sleep(delay)

        # Baseline metrics
        acoustic_density = random.randint(0, 100)
        rf_attenuation = random.uniform(0.0, 1.0)

        # Simulate crowd surge (5% chance)
        if random.random() < 0.05:
            acoustic_density = random.randint(91, 100)
            rf_attenuation = random.uniform(0.86, 1.0)

        payload = {
            "node_id": node_id,
            "node_type": node_type,
            "acoustic_density": acoustic_density,
            "rf_attenuation": rf_attenuation,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        try:
            async with session.post(TARGET_URL, json=payload) as response:
                # Discard response to avoid leaking memory or hanging
                pass
        except aiohttp.ClientError:
            # Silently ignore connection errors
            pass
        except Exception:
            # Catch other potential errors so the task loop doesn't crash
            pass

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # Spawn gate nodes
        for i in range(1, NUM_GATE_NODES + 1):
            tasks.append(asyncio.create_task(run_node(f"gate_{i}", "gate", session)))
            
        # Spawn food_stall nodes
        for i in range(1, NUM_FOOD_STALL_NODES + 1):
            tasks.append(asyncio.create_task(run_node(f"food_stall_{i}", "food_stall", session)))
            
        # Spawn bathroom nodes
        for i in range(1, NUM_BATHROOM_NODES + 1):
            tasks.append(asyncio.create_task(run_node(f"bathroom_{i}", "bathroom", session)))
            
        # Run all nodes concurrently
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    print(f"Starting edge simulator with {NUM_GATE_NODES + NUM_FOOD_STALL_NODES + NUM_BATHROOM_NODES} nodes...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSimulator stopped.")
