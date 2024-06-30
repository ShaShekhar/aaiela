## AAIELA: AI Assisted Image Editing with Language and Audio

This project empowers users to modify images using just audio commands.

By leveraging open-source AI models for computer vision, speech-to-text, large language models (LLMs),
and text-to-image inpainting, we have created a seamless editing experience that bridges the gap between
spoken language and visual transformation.


https://github.com/ShaShekhar/aaiela/assets/36410320/cc267599-beb9-4451-8c8b-138e223291f4

## Project Structure

- **detectron2**: The Detectron2 library for object detection, keypoint detection, instance/panoptic segmentation etc.
- **faster_whisper**: Contains the faster_whisper which is implementation of OpenAI Whisper for audio transcription/translation.
- **language_model**: Using small Language model like Phi3 or any of the LLM API: Gemini, Claude, GPT4 etc to extract object, action and prompt from natural language instruction.
- **sd_inpainting**: Include Text conditioned Stable Diffusion v1.5 Inpainting model.

## Installation:

See [installation instructions](INSTALL.md).

API Keys: Create a `.env` file in the root directory of the project. Fill in API keys if intend to use API-based
language models. Use the provided `.env.example` file as a template.

Or to use a small language model like Phi-3, set the `active_model:local` in config file.

To run individual test files:

```bash
$ python -m tests.<test_file_name>
```

Configuration: adjust some settings in the `aaiela.yaml` config file e.g., device, active_model.
Toggle between using an API-based model or a local LLM by modifying the `active_model` parameter.

- Run the project's main script to load the model and start the web interface.

  `python app.py`

## Project Workflow

1. Upload: User uploads an image.
2. Segmentation: Detectron2 performs segmentation.
3. Audio Input: User records an audio command (e.g., "Replace the sky with a starry night.").
4. Transcription: Faster Whisper transcribes the audio into text.
5. Language Understanding: The LLM (Gemini, GPT4, Phi3 etc.) to extracts object, action, and prompt from the text.
6. Image Inpainting:
   - Relevant masks are selected from the segmentation results.
   - Stable Diffusion Inpainting apply the desired changes.
7. Output: The inpainted image.

## Research

1. The SDXL-Inpainting model requires retraining on a substantially larger dataset to achieve satisfactory results. The current model trained by HuggingFace shows limitations.

2. context aware automatic mask generation for prompt like this "Add a cat sitting on the wooden chair." Incorporate domain knowledge or external knowledge bases (e.g., object attributes, spatial relationships) to guide mask generation.

3. 'Segment Anything' model that could generate masks from text input was explored in research paper. [This remains an active area of research](https://github.com/IDEA-Research/Grounded-Segment-Anything).

4. Contextual Reasoning: Understand relationships between objects and actions (e.g., "sitting" implies the cat should be on top of the chair).

5. Multi-Object Mask generation: "Put a cowboy hat on the person in the right and a red scarf around their neck."

6. Integrate Visual Language model such as BLIP, to provide another layer of interaction for the users.
   - If a voice command is unclear or ambiguous, the VLM can analyze the image and offer
     suggestions or ask clarifying questions.
   - The VLM can suggest adjustments to numerical parameters based on the image content.
     etc.

## Todo

- [ ] The current [TensorRT integration for Stable Diffusion models](https://github.com/NVIDIA/TensorRT/tree/release/10.0/demo/Diffusion) lacks a working example of the text-to-image inpainting pipeline.

- [ ] Integrate ControlNet conditioned on keypoints, depth, input scribbles, and other modalities.

- [ ] Integrate Mediapipe Face Mesh to enable facial landmark detection, face geometry estimation,
      eye tracking, and other features for modifying facial features in response to audio commands (e.g., "Make me smile," "Change my eye color").

- [ ] Integrate pose landmark detection capabilities.
- [ ] Incorporate a super-resolution model for image upscaling.
- [ ] Implement interactive mask editing using Segment Anything with simple click-based interactions followed by inpainting using audio instructions.
