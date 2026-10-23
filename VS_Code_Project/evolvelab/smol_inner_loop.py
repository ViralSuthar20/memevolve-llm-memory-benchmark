import os
import time
from typing import Dict, List
from smolagents import CodeAgent, InferenceClientModel, Tool
from evolvelab.base import MemoryRequest, TrajectoryData
from evolvelab.memevolve_provider import MemEvolveProvider

class MemoryRetrievalTool(Tool):
    """
    Wraps the MemEvolve architecture into a callable tool for the SmolAgent.
    Maps to the Retrieve (R) module during active task execution.
    """
    name = "memory_retriever"
    description = "Retrieves past experiences, tools, or insights from the agent's memory database to help solve the current task."
    inputs = {'query': {'type': 'string', 'description': 'The exact task or problem you need help with.'}}
    output_type = "string"

    def __init__(self, provider: MemEvolveProvider):
        super().__init__()
        self.provider = provider

    def forward(self, query: str) -> str:
        req = MemoryRequest(query=query, context="Active Task Execution", status={})
        response = self.provider.provide_memory(req)
        
        if not response.items:
            return "No relevant memory found in the database."
        
        # Format memory items for the LLM context window
        return "\n".join([f"- {item.metadata.get('source', 'memory')}: {item.content}" for item in response.items])


class SmolAgentEvaluator:
    """
    The True Inner Loop: Executes a live SmolAgent, tracks its trajectory, 
    updates the MemEvolve database, and calculates Outer Loop metrics.
    """
    def __init__(self, memory_provider: MemEvolveProvider):
        self.provider = memory_provider
        self.provider.initialize()
        
        # Initialize the custom memory tool
        self.memory_tool = MemoryRetrievalTool(self.provider)
        
        # Initialize the LLM (Uses HuggingFace Serverless Inference API by default)
        # Ensure HF_TOKEN is set in your environment variables
        self.model = InferenceClientModel() 
        
        # Instantiate the agent with our memory tool
        self.agent = CodeAgent(tools=[self.memory_tool], model=self.model, add_base_tools=True)

    def run_benchmark(self, dataset: List[Dict]) -> Dict:
        start_time = time.time()
        total_cost = 0.0
        successful_tasks = 0
        execution_logs = []

        print(f"--- Starting SmolAgent Evaluation: {self.provider.get_memory_type()} ---")

        for task in dataset:
            print(f"\n[Executing Task]: {task['query']}")
            try:
                # 1. Agent Execution (Inner Loop Active Phase)
                self.agent.memory.steps.clear() # Reset short-term context per task
                result = self.agent.run(task["query"])
                
                status = "pass" if result else "fail"
                if status == "pass": 
                    successful_tasks += 1
                
                # 2. Trajectory Extraction
                # Pull the raw reasoning steps from SmolAgent's internal state
                raw_steps = self.agent.memory.steps
                trajectory_logs = [{"action": str(step)} for step in raw_steps]
                
                # 3. Memory Ingestion (Encode & Store Modules)
                traj_data = TrajectoryData(
                    query=task["query"],
                    trajectory=trajectory_logs,
                    metadata={"status": status, "final_result": str(result)}
                )
                self.provider.take_in_memory(traj_data)
                
                # 4. Metric Calculation
                # Smolagents does not natively track exact token costs for free HF models,
                # so we estimate API cost based on execution step count for the meta-evolver.
                step_cost = len(raw_steps) * 0.01 
                total_cost += step_cost
                
                execution_logs.append({"task": task["query"], "status": status, "cost": step_cost})
                
            except Exception as e:
                print(f"[Task Failed]: {e}")
                execution_logs.append({"task": task["query"], "status": "fail", "cost": 0.05})
                total_cost += 0.05

        # 5. Compile Profile for Meta-Evolver (Outer Loop)
        end_time = time.time()
        performance = (successful_tasks / len(dataset)) * 100
        delay = end_time - start_time

        profile = {
            "id": self.provider.get_memory_type(),
            "performance": round(performance, 2),
            "cost": round(total_cost, 4),
            "delay": round(delay, 2),
            "logs": execution_logs
        }
        
        print(f"\nEvaluation Complete | Perf: {profile['performance']}% | Cost: ${profile['cost']} | Delay: {profile['delay']}s\n")
        return profile

# ==========================================
# Execution Test Block
# ==========================================
if __name__ == "__main__":
    # Ensure Hugging Face token is available
    if not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN environment variable not set. The HfApiModel will likely fail.")
    
    # Simple real-world tasks to test LLM reasoning and memory lookup
    test_dataset = [
        {"query": "What is the capital of France? Multiply the number of letters in the capital's name by 5."},
        {"query": "If I have 3 apples and eat 1, how many are left?"}
    ]

    provider = MemEvolveProvider()
    evaluator = SmolAgentEvaluator(provider)
    
    # Run the live agent
    candidate_profile = evaluator.run_benchmark(test_dataset)
    
    print("Output Profile ready for Meta-Evolver:")
    print(candidate_profile)