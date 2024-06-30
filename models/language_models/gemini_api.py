import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
import json
import re
import os

class Gemini:
    def __init__(self, api_key=None, model_name="gemini-pro"):
        if api_key is None:
            load_dotenv(find_dotenv())

            api_key=os.getenv("GEMINI_API_KEY")

        genai.configure(api_key=api_key)
        
        if not api_key:
            raise ValueError(
                "GEMINI API key not found. "
                "Please set the GEMINI_API_KEY environment variable."
            )
        self.model = genai.GenerativeModel(model_name)
    
    def list_model(self):
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)

    def getJson(self, user_prompt):
        query = """
        Task: Transform a natural language image editing instruction into a series of JSON objects for use with Stable Diffusion.
        Input Format: A sentence or paragraph describing one or more editing steps, possibly including the conjunctions "and" or "then".

        Output Format:
            - If the input is valid, a JSON array containing objects with keys "action", "object", and "prompt" (see below).
            - If the input is empty or cannot be parsed, an empty JSON array (`[]`).

        Object Structure: A JSON array where each element is a JSON object with the following fields:
            - "action": A string representing the type of image manipulation 
                (e.g., "replace", "add", "stylize", "transform", "recolor", "remove", "inpaint").
            - "object": A string or an array of strings identifying the object(s) to be manipulated (e.g., "person", "background", "car").
            - "prompt": A short description include visual details (colors, textures, styles), within the prompt for guiding stable diffusion inpainting model effectively.
                        Omit the action word (e.g., "replace", "stylize") from the prompt.

        Example:
        Input: "Stylize the car with a cyberpunk aesthetic."
        Output: [{{"action": "stylize", "object": "car", "prompt": "the car with a cyberpunk aesthetic, featuring neon accents, chrome details, and futuristic modifications."}}]
        Input: "Replace the person with statue or sculpture complementing the architecture."
        Output: [{{ "action": "replace", "object": "person", "prompt": "a statue or sculpture that seamlessly integrates with the existing architecture, consider the style, materials of the surrounding structures."}}]

        Input: {}
        Output: 
        """.format(user_prompt)
        # (e.g., "replace", "add", "stylize", "transform", "recolor", "remove", "blend", "inpaint", "outpaint", "relight", "upscale/downscale", "denoise", "restore", "depth_manipulation").
        # - "additional_attributes": (Optional) A dictionary containing additional attributes relevant to the action (e.g., "color", "intensity", "style", "mask", "object_id", "coordinates").
        # Input: "Add a fluffy cat wearing a red bow tie sitting on the chair."
        # Output: [{"action": "add", "object": "chair", "prompt": "A fluffy cat with white fur and a red bow tie sitting on a wooden chair in the corner of the room."}]
        # Input: "Stylize the car with a cyberpunk aesthetic, then change the background to a neon-lit cityscape at night."
        # Output:
        # [{"action": "stylize", "object": "car", "prompt": "the car with a cyberpunk aesthetic, featuring neon accents, chrome details, and futuristic modifications."},
        # {"action": "replace", "object": "background", "prompt": "A vibrant, neon-lit cityscape at night, with towering skyscrapers, glowing advertisements, and streaks of light from passing vehicles."}]

        # If the action doesn't apply to a specific object, leave this field blank.
        # If the action doesn't require a prompt, leave this field blank.
        try:
            response = self.model.generate_content(query)
            resp_text = response.text
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
    user_prompt = "Replace the sky with a sunset."
    model = Gemini()
    text, done = model.getJson(user_prompt)
    if done:
        print(text)
    else:
        print(f"Error: {text}")