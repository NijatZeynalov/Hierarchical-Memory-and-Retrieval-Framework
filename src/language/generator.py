from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LanguageGenerator:
    """Generates natural language descriptions from memory contents."""

    def __init__(
            self,
            model_name: str = "gpt2",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            max_length: int = 100
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device
        self.max_length = max_length

    def generate_description(
            self,
            memory_contents: List[Dict],
            task: str = "describe"
    ) -> str:
        """Generate description from memory contents."""
        # Create prompt
        prompt = self._create_prompt(memory_contents, task)

        # Generate text
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _create_prompt(
            self,
            contents: List[Dict],
            task: str
    ) -> str:
        """Create prompt for generation."""
        if task == "describe":
            template = "Based on the observations, describe the environment:\n"
        elif task == "navigate":
            template = "Given the environment, suggest navigation instructions:\n"
        else:
            template = f"Based on the observations, {task}:\n"

        # Add memory contents
        for item in contents:
            template += f"- {item.get('observation', '')}\n"

        return template