import torch
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
torch.random.manual_seed(0)

INSTRUCTION = """
Task: Transform a natural language image editing instruction into a series of JSON objects for use with Stable Diffusion.
Input Format: A sentence or paragraph describing one or more editing steps, possibly including the conjunctions "and" or "then".
Output Format:
    - If the input is valid, a JSON array containing objects with keys "action", "object", and "prompt" (see below).
    - If the input is empty or cannot be parsed, an empty JSON array (`[]`).
Object Structure: A JSON array where each element is a JSON object with the following fields:
    - "action": A string representing the type of image manipulation 
            (e.g., "replace", "add", "stylize", "transform", "recolor", "remove", "inpaint").
    - "object": A string or an array of strings identifying the object(s) to be manipulated (e.g., "person", "background", "car"). 
    - "prompt": A detailed, short description include visual details (colors, textures, styles), withing the prompt for guiding stable diffusion inpainting model effectively.
                Omit the action word (e.g., "replace", "stylize") from the prompt.

Input: "Stylize the car with a cyberpunk aesthetic."
Output: [{"action": "stylize", "object": "car", "prompt": "the car with a cyberpunk aesthetic, featuring neon accents, chrome details, and futuristic modifications."}]
Input: "Replace the person with statue or sculpture complementing the architecture."
"""
OUTPUT = """[{"action": "replace", "object": "person", "prompt": "a statue or sculpture that seamlessly integrates with the existing architecture, consider the style, materials of the surrounding structures."}]"""


class Phi3_mini_4k_instruct:
    def __init__(self, device):
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct", 
            device_map=device,
            torch_dtype="auto", 
            trust_remote_code=True,
            cache_dir='/home/shashank/Documents/AI-ML/aaiela/weights'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            cache_dir='/home/shashank/Documents/AI-ML/aaiela/weights'
        )

    def getJson(self, user_prompt):
        messages = [
            {"role": "user", "content": INSTRUCTION},
            {"role": "assistant", "content": OUTPUT},
            {"role": "user", "content": user_prompt},
        ]

        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        try:
            output = pipe(messages, **generation_args)
            resp_text = output[0]['generated_text']
            # print(resp_text)
            norm_json, flag = self.normalize_json_output(resp_text)
            response = norm_json, flag
        except Exception as e:
            response = str(e), False
        
        return response
    
    def normalize_json_output(self, raw_output):
        """
        Normalizes the raw output from a LLM into a valid JSON array.
        Args:
            raw_output (str): The raw output string from the AI model.
        """
        # Remove backticks, "json" prefix
        cleaned_json = re.sub(r'^```json\s*|\s*```', '', raw_output)
        cleaned_json = cleaned_json.replace("\n", "")
        # Pattern to match prompts with any number of separated double quotes
        pattern = r'"prompt":\s*"(.*?)"((\s*".*?")*)'

        def replace_prompt(match):
            first_part = match.group(1) 
            remaining_parts = re.findall(r'"(.*?)"', match.group(2))
            # Preserve spaces between parts
            all_parts = [first_part] + remaining_parts
            cleaned_parts = [part.strip() for part in all_parts if part.strip()]
            return '"prompt": " ' + " ".join(cleaned_parts) + ' "'
        
        # Perform the replacement using re.sub
        json_str = re.sub(pattern, replace_prompt, cleaned_json, flags=re.DOTALL)
        try:
            output_data = json.loads(json_str)
            return output_data, True
        except json.JSONDecodeError:
            return "invalid", False  # Invalid JSON, return an empty array


if __name__ == "__main__":
    phi3_model = Phi3_mini_4k_instruct(device='cpu')
    # user_prompt = "Stylize the car with a cyberpunk aesthetic, then change the background to a neon-lit cityscape at night."
    # print(phi3_model.generated_text(user_prompt))

    user_prompt = "Replace the sky with a sunset."
    resp, flag = phi3_model.getJson(user_prompt)
    print(resp)

    # user_prompt = "Add a fluffy cat wearing a red bow tie sitting on the chair."
    # print(phi3_model.generated_text(user_prompt))