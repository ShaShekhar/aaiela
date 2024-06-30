import numpy as np
import pickle
import logging
import os
import math
from PIL import Image
import importlib
from omegaconf import OmegaConf
from dotenv import load_dotenv

from models import detectron   # detectron2 model
from models.faster_whisper import WhisperModel # audio transcription model
from models.sd_inpaint import text2image_inpaint # stable diffusion inpainting model

load_dotenv()

config = OmegaConf.load("configs/aaiela.yaml")

logging.basicConfig(filename=config.LOG_FILE_PATH, format='%(asctime)s: %(levelname)s: %(message)s',
   level=logging.INFO, datefmt='%d/%m/%Y %I:%M:%S %p')

PROCESSING_COUNTER = 0

detectron_model = detectron.Detectron(**config.models.detectron2)
logging.info("Successfully loaded Detectron Model!")

whisper_model = WhisperModel(**config.models.faster_whisper)
logging.info("Successfully loaded Whisper Model!")

llm_type = config.models.language_model.active_model

if llm_type == "local":
    local_attrib = config.models.language_model.local
    local_model = local_attrib.model_name # phi3
    logging.info(f"Using local language model: {local_model}")

    local_module = importlib.import_module(f"models.language_models.{local_model}")
    model_class = getattr(local_module, local_attrib.model_class)
    language_model = model_class(local_attrib.device)
else:
    provider = config.models.language_model.api.provider
    logging.info(f"Using {provider} language model API")
    if provider == 'gemini':
        api_key = os.getenv("GEMINI_API_KEY")
    elif provider == 'openai':
        api_key = os.getenv("OPENAI_API_KEY")
    # Dynamically import the provider module
    provider_module = importlib.import_module(f"models.language_models.{provider}_api")

    # Get the model class from the provider module
    model_class = getattr(provider_module, config.models.language_model.api.provider.capitalize())

    # Initialize and use the language model
    model_name = config.models.language_model.api[f"{provider}"].model_name  
    language_model = model_class(api_key, model_name=model_name)

# Loading stable diffusion inpainting model.
sd_param = config.models.sd_inpaint
sd_model = text2image_inpaint.SdInpainting(sd_param.path, sd_param.ckpt, sd_param.device)
logging.info("Successfully Loaded SD Inpainting Model.")


# segment anything
# from models import segmentAnything as segment  # segmentation model
segment = None # enable import
def detect_and_segment(imgPath, imgName, index=0):
    pil_image = Image.open(imgPath).convert("BGR")
    img_array = np.array(pil_image)

    detect_img, bboxes, labels = detectron_model.predict(img_array)
    # if no bbox detected then just run the automatic maskgenerator
    if len(bboxes):
        img_masked, mask_arrays = segment.predict(img_array, bboxes)
    else:
        img_masked, mask_arrays = segment.predict(img_array)
    
    # Save processed image (optional, depending on your needs)
    img_folder = imgName.split('_')[1].split('.')[0]
    processed_filename = 'segment_' + imgName
    processed_filepath = os.path.join(config['UPLOAD_FOLDER'], img_folder, processed_filename)
    pil_img_masked = Image.fromarray(img_masked)
    pil_img_masked.save(processed_filepath)

    # save all the metadata
    metadata_path = os.path.join(config['UPLOAD_FOLDER'], img_folder + '_meta.pkl')
    with open(metadata_path, "wb") as f:
        pickle.dump((imgName, labels, bboxes, mask_arrays, index), f)
    
    return processed_filepath

def get_image_orientation_and_ar(pil_image, imgPath):
    # save aspect ratio adjusted resized image.
    width, height = pil_image.size
    gcd = math.gcd(width, height) # Find the greatest common divisor
    simplified_width = width // gcd
    simplified_height = height // gcd

    aspect_ratio = simplified_width / simplified_height 
    landscape = 1 # assume it's a Landscape.
    # Determine the new height based on the new width and calculated aspect ratio
    if simplified_width > simplified_height:  # Landscape
        new_height = config.IMG_SIZE
        new_width = int(new_height * aspect_ratio)
    else:  # Portrait (or Square)
        new_width = config.IMG_SIZE
        new_height = int(new_width / aspect_ratio)
        landscape = 0
    
    resize_img = pil_image.resize((new_width, new_height))
    resize_img.save(imgPath)

    return resize_img, landscape, aspect_ratio

def panoptic_segment(imgPath, imgName):
    global PROCESSING_COUNTER
    PROCESSING_COUNTER = 0
    pil_image = Image.open(imgPath).convert("RGB")
    resize_img, landscape, ar = get_image_orientation_and_ar(pil_image, imgPath)

    try:
        img_array = np.array(resize_img)[:, :, ::-1] # detectron input: BGR
        segmented_img, mask_array, labels = detectron_model.predict(img_array)
        # Save processed image
        img_dir_name = imgName.split('_')[1].split('.')[0]
        processed_filename = 'segment_' + imgName
        segmented_imgPath = os.path.join(config['UPLOAD_FOLDER'], img_dir_name, processed_filename)
        pil_img_masked = Image.fromarray(segmented_img)
        pil_img_masked.save(segmented_imgPath)

        logging.info("Panoptic Segmentation done.")

        # save all the metadata
        metadata_path = os.path.join(config['UPLOAD_FOLDER'], img_dir_name + '_meta.pkl')
        with open(metadata_path, "wb") as f:
            pickle.dump((imgName, labels, mask_array, landscape, ar), f)
        
        logging.info("Saved metadata pickle file!")

    except Exception as e:
        logging.warning(str(e))
        return str(e), False

    return segmented_imgPath, True

def transcribe(audio_data):
    # process audio data
    try:
        segments, info = whisper_model.transcribe(audio_data)
        seg = []
        for segment in segments:
            seg.append(segment.text)
        text_data = ''.join(seg) # transcribed_text
        flag = True
        if not len(text_data):
            text_data = "No speech detected, Try Speaking again."
            flag = False
        logging.info("Whisper transcription done.")

        return text_data, flag
    except Exception as e:
        error = "Error during Transcription, Please try again."
        logging.warning(error)
        return error, False


def text_cond_inpainting(img_name, img_dir, img_path, mask_uint8, prompt, landscape, ar):
    global PROCESSING_COUNTER

    logging.info("SD-Inpainting processing.")
    img_inpaint = sd_model.main(img_path, mask_uint8, prompt,
                        sd_param.ddim_steps, sd_param.ddim_eta, sd_param.scale)
                
    logging.info("SD-Inpainting processing done.")
    PROCESSING_COUNTER += 1
    imgName = str(PROCESSING_COUNTER) + '_' + img_name.split('_')[1]
    img_path = os.path.join(config['UPLOAD_FOLDER'], img_dir, imgName) # assign img_path

    pil_image = Image.fromarray(img_inpaint)

    if landscape:
        new_height = config.IMG_SIZE
        new_width = int(new_height * ar)
    else:
        new_width = config.IMG_SIZE
        new_height = int(new_width / ar)
    resize_img = pil_image.resize((new_width, new_height))
    resize_img.save(img_path)

    return img_path

def instruction_to_image_pipeline(imgName, text_data):
    img_dir = imgName.split('.')[0] # 'test'
    pkl_path = os.path.join(config['UPLOAD_FOLDER'], img_dir)
    
    with open(pkl_path + '_meta.pkl', "rb") as f:
        img_name, labels, mask_arrays, landscape, ar = pickle.load(f)
    
    if PROCESSING_COUNTER:
        imgName = str(PROCESSING_COUNTER) + '_' + img_name.split('_')[1]
        img_path = os.path.join(config['UPLOAD_FOLDER'], img_dir, imgName)
    else:
        img_path = os.path.join(config['UPLOAD_FOLDER'], img_dir, img_name)

    # extract the action, obj_name and prompt to inpaint the image
    obj_list, flag = language_model.getJson(text_data)
    resp_msg = None

    if flag and len(obj_list):
        logging.info(f"LLM Response: {obj_list}")
        not_in_lables = [] # requested obj not in labels.
        processed = 0

        for each_obj in obj_list:
            try:
                # ("replace", "add", "stylize", "transform", "recolor", "remove", "inpaint").
                action = each_obj["action"].lower()
                obj = each_obj["object"].lower()
                prompt = each_obj["prompt"] # prompt might be empty
            except Exception as e:
                resp_msg = str(e)
                continue

            if action in ('add'):
                # get the mask from the user then apply text cond inpainting
                resp_msg = f"The action '{action}' is not currently supported."
                continue
            elif action in ('resize'):
                resp_msg = f"The action '{action}' is not currently supported."
                continue
            
            if obj in labels:
                idx = labels.index(obj)
                # get the mask at that idx
                mask_uint8 = mask_arrays[idx]

                if action in ('replace', 'inpaint', 'recolor', 'transform'):
                    # call text condition inpainting model
                   img_path = text_cond_inpainting(img_name, img_dir, img_path, mask_uint8, prompt, landscape, ar)
                   processed += 1
                elif action in ('remove'):
                    # inpainting model
                    resp_msg = f"The action '{action}' is not currently supported."
                elif action in ('stylize'):
                    # controlnet model
                    resp_msg = f"The action '{action}' is not currently supported."
                else:
                    resp_msg = f"The action '{action}' is not currently supported."
            else:
                not_in_lables.append(obj)
        
        if len(not_in_lables) and (processed == 0):
            logging.info(f"Requested {not_in_lables}, not in labels.")
            resp_msg = f"Requested {not_in_lables}, not in lables."
            return resp_msg, False
        elif resp_msg is not None:
            logging.info(resp_msg)
            return resp_msg, False
        elif processed >= 1:
            return img_path, True
    else:
        logging.info(f"LLM Response: {obj_list}")
        error = 'LLM failed to parse, Please try again.'
        logging.warning(error)
        return error, False
        