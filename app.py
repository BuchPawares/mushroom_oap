from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import gdown
import json

app = Flask(__name__)
model_path = "mushroom_model1.h5"
if not os.path.exists(model_path):
    # Google Drive file ID
    file_id = "18ggGp7TjmmzGUP9PDJvDQlZnh4DRxcb5"  # ← เปลี่ยนเป็น ID จริงจากลิงก์ GDrive
    url = f"https://drive.google.com/uc?id={file_id}"
    print("Downloading model...")
    gdown.download(url, model_path, quiet=False)

# Load model
model = load_model(model_path)

# โหลดชื่อ class
with open("class_names.json", "r") as f:
    class_names = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        file = request.files['file']
        img = image.load_img(file.stream, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

    # ไม่ต้องหาร 255 ถ้าในโมเดลมี Rescaling(1./255)
        predictions = model.predict(img_array)
    
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = tf.nn.softmax(predictions[0])[predicted_index].numpy()
        return jsonify({
            'label': predicted_label,
            'confidence': round(confidence, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# สำหรับโฮสต์บน Render ต้องระบุ host และ port
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
