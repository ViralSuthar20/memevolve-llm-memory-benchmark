import time
import random
from typing import Dict
from base import MemoryRequest, TrajectoryData
from memevolve_provider import MemEvolveProvider

class InnerLoopEvaluator:
    """
    The Inner Loop of MemEvolve: Executes tasks, utilizes the memory system, 
    and calculates Performance, Cost, and Delay.
    """
    def __init__(self, memory_provider):
        self.memory = memory_provider
        self.memory.initialize()

    def run_benchmark(self, dataset: list) -> Dict:
        """
        Simulates an agent running through a dataset (e.g., TaskCraft or GAIA).
        Returns the evaluation profile for the Outer Loop.
        """
        start_time = time.time()
        total_cost = 0.0
        successful_tasks = 0
        execution_logs = []

        print(f"--- Starting Inner Loop Evaluation: {self.memory.get_memory_type()} ---")

        for task in dataset:
            # 1. Agent requests memory for the current task
            req = MemoryRequest(query=task["query"], context="Task Execution", status={"step": 1})
            retrieved_mem = self.memory.provide_memory(req)
            
            # 2. Simulate Agent Execution (Pass/Fail based on memory retrieval)
            # In a real setup, this is where SmolAgent or Flash-Searcher runs.
            # Here, we simulate success if relevant memory was retrieved.
            if retrieved_mem.items or random.random() > 0.5:
                status = "pass"
                successful_tasks += 1
            else:
                status = "fail"

            # 3. Agent encodes and stores the trajectory
            trajectory = TrajectoryData(
                query=task["query"],
                trajectory=[{"action": f"Action 1 for {task['query']}"}, {"action": f"Action 2 for {task['query']}"}],
                metadata={"status": status}
            )
            self.memory.take_in_memory(trajectory)

            # 4. Log metrics
            step_cost = random.uniform(0.01, 0.05) # Simulated API cost
            total_cost += step_cost
            execution_logs.append({"task": task["query"], "status": status, "cost": step_cost})

        # Calculate final metrics
        end_time = time.time()
        performance = (successful_tasks / len(dataset)) * 100
        delay = end_time - start_time

        profile = {
            "id": self.memory.get_memory_type(),
            "performance": round(performance, 2),
            "cost": round(total_cost, 4),
            "delay": round(delay, 2),
            "logs": execution_logs
        }
        
        print(f"Evaluation Complete | Perf: {profile['performance']}% | Cost: ${profile['cost']} | Delay: {profile['delay']}s\n")
        return profile

# ==========================================
# Execution Test Block
# ==========================================
if __name__ == "__main__":
    # 1. Define a mock dataset (Subset of TaskCraft)
    mock_taskcraft = [
        {"query": "Find the release date of the 2019 BAFTA game winner"},
        {"query": "Calculate the trajectory of a projectile"},
        {"query": "Summarize the latest AI research paper"},
        {"query": "Find the release date of the 2019 BAFTA game winner"} # Repeated to test memory
    ]

    # 2. Instantiate our Baseline Memory Provider
    provider = MemEvolveProvider()

    # 3. Run the Inner Loop
    evaluator = InnerLoopEvaluator(provider)
    candidate_profile = evaluator.run_benchmark(mock_taskcraft)
    
    # 4. (Next Step Integration) This candidate_profile is what you feed into meta_evolver.py!
    print("Output Profile ready for Meta-Evolver:")
    print(candidate_profile)