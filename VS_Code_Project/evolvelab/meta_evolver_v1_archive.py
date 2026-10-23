import os
import re
from typing import List, Dict
from huggingface_hub import InferenceClient

class MetaEvolver:
    """
    The Outer Loop of MemEvolve: Evaluates memory candidates and rewrites their architecture code.
    This corresponds to the Diagnose-and-Design module described in the paper.
    """
    def __init__(self):
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN missing. Cannot initialize Meta-Evolver.")
        
        # Using a highly capable instruction-tuned model for code generation
        self.client = InferenceClient(model="meta-llama/Llama-3.3-70B-Instruct", token=token)

    def evaluate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Pareto-ranking based on the tuple: F = (Performance, -Cost, -Delay)"""
        ranked = sorted(
            candidates, 
            key=lambda x: (x.get('performance', 0), -x.get('cost', float('inf')), -x.get('delay', float('inf'))), 
            reverse=True
        )
        return ranked

    def diagnose(self, execution_logs: List[Dict]) -> str:
        """Analyzes inner-loop execution logs to identify defects."""
        print("   -> Running LLM Diagnosis...")
        prompt = (
            "You are a Meta-Evolver diagnosing an LLM agent's memory architecture.\n"
            "Analyze these execution logs and identify specific defects in the Encode, "
            "Store, Retrieve, or Manage functions.\n"
            f"Logs: {execution_logs}\n\n"
            "Keep your diagnosis concise and actionable (under 4 sentences)."
        )
        
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return response.choices[0].message.content

    def design(self, current_code: str, diagnosis: str) -> str:
        """Rewrites the architecture code based on the diagnosis report."""
        print("   -> Running LLM Code Generation (This may take 30-60 seconds)...")
        prompt = (
            "You are an expert Python engineer. Rewrite the following MemEvolveProvider class "
            "to fix the architectural defects identified in the diagnosis. \n"
            "CRITICAL: Output ONLY valid Python code inside a ```python ``` codeblock. "
            "Do not include any conversational text, explanations, or markdown outside the codeblock.\n\n"
            f"Diagnosis:\n{diagnosis}\n\n"
            f"Current Code:\n{current_code}"
        )
        
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2500
        )
        content = response.choices[0].message.content
        
        # Extract purely the Python code from the LLM's markdown formatting
        match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
        if match:
            return match.group(1)
        
        # Fallback if the LLM forgets the python specifier
        match_generic = re.search(r'```\n(.*?)\n```', content, re.DOTALL)
        if match_generic:
            return match_generic.group(1)
            
        return content.strip()

    def run_evolution_cycle(self, candidate_profiles: List[Dict], provider_filepath: str, output_filepath: str):
        if not candidate_profiles:
            raise ValueError("Candidate profiles list cannot be empty.")
            
        # 1. Selection
        ranked_candidates = self.evaluate_candidates(candidate_profiles)
        top_parent = ranked_candidates[0]
        print(f"Selected Top Architecture with Performance: {top_parent.get('performance')}%")
        
        with open(provider_filepath, "r") as f:
            current_code = f.read()
            
        # 2. Diagnosis
        diagnosis_report = self.diagnose(top_parent.get('logs', []))
        print(f"\n--- Diagnosis Report ---\n{diagnosis_report}\n")
        
        # 3. Design
        new_architecture_code = self.design(current_code, diagnosis_report)
        
        with open(output_filepath, "w") as f:
            f.write(new_architecture_code)