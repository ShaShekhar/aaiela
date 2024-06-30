from models.language_models import gemini_api

user_prompt = "Replace the sky with a sunset."
model = gemini_api.Gemini()
text, done = model.getJson(user_prompt)
if done:
    print('Test Passed!')
    print(text)
else:
    print('Test Failed!')
    print(f"Error: {text}")