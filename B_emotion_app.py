from flask import Flask, render_template, request
import numpy as np
import os
import base64
from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

app = Flask(__name__)

# ============================
# 🔥 REBUILD MODEL
# ============================
base_model = MobileNetV2(
    weights=None,
    include_top=False,
    input_shape=(224,224,3)
)

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

# ============================
# 🔥 LOAD WEIGHTS
# ============================
model.load_weights("model.weights.h5")

print("✅ Model loaded successfully!")

# ============================
# 😊 EMOTION LABELS
# ============================
emotion_map = {
    0:'Angry 😠',
    1:'Disgust 🤢',
    2:'Fear 😨',
    3:'Happy 😄',
    4:'Sad 😢',
    5:'Surprise 😲',
    6:'Neutral 😐'
}

# ============================
# 📁 UPLOAD FOLDER
# ============================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================
# 🔥 PREDICTION FUNCTION
# ============================
def predict_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224,224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)

    pred_class = np.argmax(pred)
    confidence = np.max(pred)

    return emotion_map[pred_class]

# ============================
# 🌐 ROUTES
# ============================
# @app.route("/", methods=["GET", "POST"])
# def index():
    # prediction = None
# 
    # if request.method == "POST":
        # file = request.files["file"]
# 
        # if file:
            # filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            # file.save(filepath)
# 
            # prediction = predict_image(filepath)
# 
    # return render_template("index1.html", prediction=prediction)
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":

        # =========================
        # ✅ OPTION 1: FILE UPLOAD
        # =========================
        if "file" in request.files:
            file = request.files["file"]

            if file and file.filename != "":
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)

                prediction = predict_image(filepath)

        # =========================
        # ✅ OPTION 2: CAMERA IMAGE
        # =========================
        elif "image_data" in request.form:
            image_data = request.form["image_data"]

            if image_data:
                image_data = image_data.split(",")[1]
                image_bytes = base64.b64decode(image_data)

                filepath = os.path.join(UPLOAD_FOLDER, "camera_capture.png")

                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                prediction = predict_image(filepath)

    return render_template("index.html", prediction=prediction)
# ============================
# 🚀 RUN
# ============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)