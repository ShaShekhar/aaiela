from models.faster_whisper import WhisperModel

model_size_or_path = "small.en"
weight_path = "weights"
audio_path = 'tests/assets/jfk.flac'

try:
    model = WhisperModel(model_size_or_path, device="cuda", compute_type="float16", download_root=weight_path)
    segments, info = model.transcribe(audio_path)
    seg = []
    for segment in segments:
        seg.append(segment.text)
    text_data = ''.join(seg)
    print('Test Passed!')
    print(text_data)
except Exception as e:
    print('Test Failed!')
    print(str(e))