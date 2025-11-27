from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Create Flask application
app = Flask(__name__)

# Load trained deep learning model
model = load_model("emotion_model.h5")

# Emotion categories (must match your model output)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Short positive messages for each emotion
DESCRIPTIONS = {
    "Happy": "Keep smiling! Share your joy today.",
    "Sad": "Don't worry, better days are ahead.",
    "Angry": "Take a deep breath and relax.",
    "Surprise": "Wow! Something unexpected!",
    "Neutral": "Calm and composed — keep going.",
    "Fear": "Stay strong. Courage conquers fear.",
    "Disgust": "Try shifting your focus to something positive."
}

# Folder to save uploaded images (optional)
UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    """
    Converts image into a format suitable for MobileNetV2:
    - Load image
    - Resize to 224x224
    - Normalize pixel values
    - Expand dimensions for model input
    """
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize((48, 48))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route("/")
def index():
    """Load the main homepage (index.html)."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload and returns the predicted emotion."""
    
    # Get uploaded file
    file = request.files["image"]

    # Save file to uploads folder
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess image
    img = preprocess_image(filepath)

    # Make model prediction
    predictions = model.predict(img)

    # Get emotion with highest probability
    emotion_idx = np.argmax(predictions)
    emotion = EMOTIONS[emotion_idx]

    # Get description text
    description = DESCRIPTIONS.get(emotion, "")

    # Return result as JSON
    return jsonify({
        "emotion": emotion,
        "description": description
    })

# Run the Flask server
# commented out because of vercel deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
