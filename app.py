from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

# โหลดโมเดล
model = load_model("mushroom_model1.h5")

# โหลดชื่อ class
class_names = ["tyer1", "tyer2"] # เปลี่ยนตามโมเดลคุณ

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        file = request.files['file']
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))  # ปรับให้ตรงกับขนาด input ของโมเดล
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_label = class_names[predicted_index]
        confidence = float(prediction[predicted_index])

        return jsonify({
            'label': predicted_label,
            'confidence': round(confidence, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# สำหรับโฮสต์บน Render ต้องระบุ host และ port
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
