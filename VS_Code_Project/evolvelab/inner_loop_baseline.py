import os
import time
from smolagents import CodeAgent, InferenceClientModel, tool
from dataset_loader import BenchmarkLoader

# ==========================================
# BASELINE ARCHITECTURE (A_0)
# This represents the static, non-evolved memory
# ==========================================
@tool
def manage_memory(query: str) -> str:
    """
    A baseline, static memory retrieval module.
    Args:
        query: The search query to lookup past context.
    """
    # In the baseline, memory is static and does not effectively retrieve dynamic context.
    return "Baseline Memory: No relevant historical context found for this query."

class InnerLoopEvaluator:
    def __init__(self):
        print("Initializing SmolAgents Inner Loop Evaluator...")
        # Utilizing the serverless inference API as defined in the report
        self.model = InferenceClientModel(
            model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            token=os.environ.get("HF_TOKEN")
        )
        self.agent = CodeAgent(tools=[manage_memory], model=self.model)
        
        # Hyperparameters from the LaTeX Section 4.2
        self.alpha = 0.6
        self.beta = 0.2
        self.gamma = 0.2

    def evaluate_baseline(self, tasks):
        print(f"Beginning evaluation on {len(tasks)} tasks...\n")
        
        results = []
        total_success = 0
        total_tokens = 0
        total_latency = 0.0

        for task in tasks:
            question = task.trajectory[0]['content']
            ground_truth = task.trajectory[0]['ground_truth']
            
            print(f"Executing Task: {task.task_id}")
            
            start_time = time.time()
            try:
                # Execute reasoning trace
                output = self.agent.run(question)
                success = 1 if ground_truth.lower() in str(output).lower() else 0
                
                # Mocking token count for baseline tracking (smolagents abstracts this heavily)
                tokens_used = len(str(self.agent.memory.steps)) * 1.5 
                
            except Exception as e:
                print(f"Agent crashed on task: {e}")
                success, output, tokens_used = 0, "ERROR", 500
                
            latency = time.time() - start_time
            
            # Record metrics
            total_success += success
            total_tokens += tokens_used
            total_latency += latency
            
            results.append({
                "task_id": task.task_id,
                "success": success,
                "tokens": tokens_used,
                "latency": latency
            })
            
            print(f"Result: {'PASS' if success else 'FAIL'} | Latency: {latency:.2f}s\n")

        # Calculate final F score components
        avg_success = total_success / len(tasks)
        avg_cost = (total_tokens / 1000) * 0.0002 # Simulated API cost
        avg_latency = total_latency / len(tasks)
        
        f_score = (self.alpha * avg_success) - (self.beta * avg_cost) - (self.gamma * avg_latency)
        
        print("==========================================")
        print("BASELINE METRICS PROFILED (A_0)")
        print("==========================================")
        print(f"Average Success Rate : {avg_success * 100}%")
        print(f"Average Latency      : {avg_latency:.2f} seconds")
        print(f"Total Token Cost     : ${avg_cost:.5f}")
        print(f"Overall Fitness (F)  : {f_score:.4f}")
        print("==========================================\n")
        
        return results

# --- Execution ---
if __name__ == "__main__":
    loader = BenchmarkLoader()
    # Pulling 2 samples to prevent immediate API rate limiting
    gaia_tasks = loader.load_gaia_validation(max_samples=2) 
    
    evaluator = InnerLoopEvaluator()
    evaluator.evaluate_baseline(gaia_tasks)