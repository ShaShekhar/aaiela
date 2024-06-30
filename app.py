from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS
import subprocess
from PIL import Image
import os
import io
import main

app = Flask(__name__, template_folder='web_frontend/templates', static_folder='web_frontend/static')
cors = CORS(app)
app.config['UPLOAD_FOLDER'] = main.config.UPLOAD_FOLDER

os.makedirs(main.config.UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', recording_time=main.config.web_frontend.recording_time)

@app.route("/process_image", methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file received!'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if allowed_file(image_file.filename):
        img_name = secure_filename(image_file.filename)
        img_name = img_name.replace('_', '') # remove '_'
        img_dir_name = img_name.split('.')[0] # 'test'
        img_folder = os.path.join(app.config['UPLOAD_FOLDER'], img_dir_name)
        os.makedirs(img_folder, exist_ok=True)
        # uploads -> imgName folder -> 0_img.jpg, 1_img.jpg etc
        img_name = '0_' + img_name # 0 indicate index
        img_path = os.path.join(img_folder, img_name)
    else:
        return jsonify({"error": "Invalid image format. Allowed formats are PNG, JPG, and JPEG."}), 400

    try:
        # Read image from in-memory file-like object
        with io.BytesIO(image_file.read()) as f:
            pil_image = Image.open(f)
            pil_image.save(img_path)
    except Exception as e:
        return jsonify({'error': 'Error reading/saving image, Please try again.'}), 500

    segImg, flag = main.panoptic_segment(img_path, img_name) # error or imgPath
    if not flag:
        return jsonify({'error': f'Error during Segmentation: {str(segImg)}'}), 500
    
    return send_file(segImg, mimetype='image/jpeg')  # Adjust mimetype if needed

def allowed_file(filename):
    # Allowed file types (customize this as needed)
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/process_text_audio", methods=['POST'])
def inpaint():
    global PROCESSING_COUNTER
    dataType = request.form.get("dataType")
    dataFile = request.files["data"]
    imgName = secure_filename(dataFile.filename) # either txt or webm
    imgName = imgName.replace('_', '') # remove '_'

    if dataType not in ["text", "audio"]:
        return jsonify({"error": "Invalid data type"}), 400

    if dataType == "audio":
        # Convert webm to raw audio bytes using FFmpeg
        cmd = [
        "ffmpeg",
        "-i", "pipe:0",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        "pipe:1"
        ]
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            audio_data, stderr = process.communicate(input=dataFile.read())
            if process.returncode != 0:
                raise Exception(f"FFmpeg error: {stderr.decode()}")
            
        except subprocess.CalledProcessError as e:
            return jsonify({"error": "FFmpeg error, Please try again.}"}), 500
        except Exception as e:
            # logging.warning(str(e))
            return jsonify({"error": "Error converting audio, Please try again."}), 500
        
        text_data, flag = main.transcribe(audio_data)
        if not flag:
            return jsonify({"error": text_data}), 400 # or 500

    elif dataType == "text":
        text_data = dataFile.read().decode("utf-8") # Decode text data

    img_or_resp, flag = main.instruction_to_image_pipeline(imgName, text_data)
    if flag:
        return send_file(img_or_resp, mimetype='image/jpeg')
    else:
        return jsonify({"error": img_or_resp}), 400

if __name__ == "__main__":
    host = os.environ.get("IP", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    app.run(host=host, port=port, debug=False)