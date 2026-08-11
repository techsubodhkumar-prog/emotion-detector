from flask import Flask, render_template, request
import numpy as np
import os
import base64
from PIL import Image
from werkzeug.utils import secure_filename

# ============================================================
# CPU ONLY
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# REBUILD MODEL
# ============================================================

base_model = MobileNetV2(
    weights=None,
    include_top=False,
    input_shape=(224, 224, 3)
)

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(7, activation="softmax")
])


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

model.load_weights("model.weights.h5")

print("Model loaded successfully!")


# ============================================================
# EMOTION LABELS
# ============================================================

emotion_map = {
    0: "Angry 😠",
    1: "Disgust 🤢",
    2: "Fear 😨",
    3: "Happy 😄",
    4: "Sad 😢",
    5: "Surprise 😲",
    6: "Neutral 😐"
}


# ============================================================
# UPLOAD FOLDER
# ============================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(path):

    img = Image.open(path).convert("RGB")

    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    pred = model.predict(
        img_array,
        verbose=0
    )

    pred_class = int(np.argmax(pred))

    confidence = float(np.max(pred))

    print(
        f"Prediction: {emotion_map[pred_class]} | "
        f"Confidence: {confidence:.2%}"
    )

    return emotion_map[pred_class]


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None

    if request.method == "POST":

        # ====================================================
        # OPTION 1:
        # CROPPED IMAGE FROM UPLOAD OR CAMERA
        # ====================================================

        if "image_data" in request.form:

            image_data = request.form["image_data"]

            if image_data:

                try:

                    # Remove data URL prefix
                    if "," in image_data:
                        image_data = image_data.split(",", 1)[1]

                    image_bytes = base64.b64decode(
                        image_data
                    )

                    filepath = os.path.join(
                        UPLOAD_FOLDER,
                        "processed_image.png"
                    )

                    with open(
                        filepath,
                        "wb"
                    ) as f:

                        f.write(image_bytes)

                    prediction = predict_image(
                        filepath
                    )

                except Exception as error:

                    print(
                        "Image processing error:",
                        error
                    )

        # ====================================================
        # OPTION 2:
        # DIRECT FILE UPLOAD
        # ====================================================

        elif "file" in request.files:

            file = request.files["file"]

            if file and file.filename != "":

                filename = secure_filename(
                    file.filename
                )

                filepath = os.path.join(
                    UPLOAD_FOLDER,
                    filename
                )

                file.save(filepath)

                prediction = predict_image(
                    filepath
                )

    return render_template(
        "index.html",
        prediction=prediction
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )