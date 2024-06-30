"""
DISCLAIMER:

This code has not undergone rigorous testing.
It may contain errors or produce unexpected results.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

class Openai:
    def __init__(self, api_key=None, model_name="gpt-3.5-turbo"):
        if api_key is None:
            load_dotenv(find_dotenv())

            api_key=os.getenv("OPENAI_API_KEY")

        self.openai_client = OpenAI(api_key=api_key)

        if not self.openai_client.api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Please set the OPENAI_API_KEY environment variable."
            )

        self.model_name = model_name
        # self.temperature = temp

    def getJson(self, prompt):
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
        """.format(prompt)

        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        return response['choices'][0]['message']['content']

if __name__ == "__main__":
    model_name = "gpt-3.5-turbo"
    model = Openai(model_name)
    response = model.toJson("Stylize the car with a cyberpunk aesthetic, then change the background to a neon-lit cityscape at night.")
    print(response)
