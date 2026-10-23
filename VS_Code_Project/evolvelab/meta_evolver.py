import os
import re
from smolagents import InferenceClientModel

class MetaEvolver:
    def __init__(self):
        print("Initializing LLM-Driven Meta-Optimizer (Llama-3.3-70B)...")
        # We use a larger model for the Outer Loop code generation
        self.meta_model = InferenceClientModel(
            model_id="meta-llama/Llama-3.3-70B-Instruct",
            token=os.environ.get("HF_TOKEN") # Use your same HF token
        )

    def diagnose_and_design(self, failed_logs: str) -> str:
        print("\n[Phase 1: Diagnosis & Design]")
        print("Feeding Inner Loop failure logs to Meta-Optimizer...")
        
        prompt = f"""
        You are a system architecture expert. An LLM agent crashed while executing the GAIA benchmark.
        
        Here are the execution logs showing the failure:
        {failed_logs}
        
        The current memory tool 'manage_memory' is a static baseline and failed. 
        Your task is to rewrite the Python code for the 'manage_memory' tool so it can dynamically search for information (e.g., scraping Wikipedia or DuckDuckGo) to answer the user's query instead of hallucinating.
        
        Requirements:
        1. Write ONLY valid Python code.
        2. Do not use the smolagents @tool decorator, just write the raw function.
        3. Name the function exactly 'evolved_memory_tool(query: str) -> str'.
        4. Wrap your code strictly inside ```python and ``` markers.
        """
        
        # Triggering the LLM to write code
        response = self.meta_model(
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content

    def extract_and_inject(self, llm_response: str):
        print("\n[Phase 2: Execution]")
        print("Extracting evolved Python code via RegEx...")
        
        # RegEx protocol as defined in Section 6.2 of the report
        match = re.search(r'```python\n(.*?)\n```', llm_response, re.DOTALL)
        
        if match:
            evolved_code = match.group(1)
            print("Successfully extracted code. Writing to disk...")
            
            # Physically mutate the architecture by overwriting a file
            with open("evolved_memory.py", "w") as f:
                f.write(evolved_code)
            print("Successfully generated 'evolved_memory.py'.")
        else:
            print("FATAL: Meta-Optimizer failed to output valid Python blocks.")

# --- Execution ---
if __name__ == "__main__":
    evolver = MetaEvolver()
    
    # Simulating the log trace from Task GAIA_val_17b5a6a3
    mock_log = "Error: 'web_search' is not allowed. Agent hallucinated 50 zip codes and failed."
    
    new_architecture = evolver.diagnose_and_design(mock_log)
    evolver.extract_and_inject(new_architecture)