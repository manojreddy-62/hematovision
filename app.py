import os
import numpy as np
# No longer directly using cv2 unless specific cv2 ops are needed outside of Keras/PIL loading.
# If you explicitly need cv2 for other reasons in the app, keep it.
# import cv2

from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf # Keep this as tf for load_model and other tf utilities

# Specific preprocessing function from MobileNetV2, as used in training
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# For image loading and conversion to array
from tensorflow.keras.preprocessing import image

# PIL.Image is implicitly used by keras.preprocessing.image, so explicit import is often not needed
# If you perform direct PIL operations, keep it.
# from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'static'  # Define a folder to temporarily store uploaded images
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Define the path to your trained model
# IMPORTANT: Based on your file explorer screenshot, your model is 'trained_model.h5'
# directly in C:\HematoVision
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model.h5")

# --- DEBUG PRINT STATEMENTS ---
print(f"DEBUG: Current working directory for app.py: {os.getcwd()}")
print(f"DEBUG: Attempting to load model from: {MODEL_PATH}")

model = None # Initialize model to None
try:
    # Use tf.keras.models.load_model which is the standard way to load models in TensorFlow 2.x
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR: Failed to load model from {MODEL_PATH}: {e}")
    print("Please ensure the model path is correct and the file exists. Check filename 'trained_model.h5'")
    # model remains None if loading fails, and the app will handle this in the routes.

# Labels must match the order during training (e.g., from ImageDataGenerator.class_indices)
CLASS_LABELS = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil'] # Renamed for clarity to CLASS_LABELS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Check if a file was uploaded as part of the request
        if "file" not in request.files:
            return render_template("home.html", message="Error: No file part in the request.")

        file = request.files["file"]

        # Check if a file was selected by the user (filename is not empty)
        if file.filename == "":
            return render_template("home.html", message="Error: No image file selected.")

        # Proceed only if a file is present and the model was loaded successfully at startup
        if file and model:
            # Create the 'static' directory if it doesn't exist.
            # os.makedirs(..., exist_ok=True) is used for robustness to prevent errors
            # if the directory already exists.
            static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), UPLOAD_FOLDER)
            os.makedirs(static_folder, exist_ok=True) 

            # Define a safe path to save the uploaded image.
            # Using original filename, but in a production app, you might use a unique ID
            # (e.g., uuid.uuid4()) to prevent overwriting files with the same name.
            img_filename = file.filename
            img_path = os.path.join(static_folder, img_filename)
            file.save(img_path) # Save the uploaded file to the static directory

            try:
                # Load and preprocess the image for prediction.
                # image.load_img handles resizing and ensures RGB format.
                img = image.load_img(img_path, target_size=(224, 224)) # Target size for MobileNetV2
                img_array = image.img_to_array(img) # Convert the PIL image to a NumPy array (H, W, C)
                img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, H, W, C)
                
                # Apply the SAME preprocessing function used during MobileNetV2 model training.
                # This normalizes pixel values from [0, 255] to [-1, 1].
                img_array = preprocess_input(img_array)

                # Make prediction using the loaded model
                predictions = model.predict(img_array)
                # For a classification model with softmax output, argmax gives the index
                # of the class with the highest predicted probability.
                predicted_class_idx = np.argmax(predictions[0]) # predictions[0] because it's a batch of 1 image
                predicted_class_label = CLASS_LABELS[predicted_class_idx]
                
                # Calculate the confidence score for the predicted class
                confidence_score = predictions[0][predicted_class_idx] * 100 # Convert to percentage

                # Construct the URL for the uploaded image in the static folder,
                # so it can be displayed in the HTML.
                img_display_url = url_for('static', filename=img_filename)

                # Render the result.html template, passing the prediction details and image URL.
                return render_template("result.html", 
                                       class_label=predicted_class_label, 
                                       confidence=f"{confidence_score:.2f}%", # Format confidence to 2 decimal places
                                       img_path=img_display_url)

            except Exception as e:
                # This 'except' block catches any errors that occur during the try block,
                # such as issues with image processing or prediction.
                
                # Clean up the temporarily uploaded file if an error occurred after saving it.
                if os.path.exists(img_path):
                    os.remove(img_path)
                
                # Provide a user-friendly error message to the home.html page.
                # We avoid passing the raw exception 'e' directly as it might contain
                # technical details that are confusing or reveal internal state.
                user_message = f"An error occurred while processing your image. Please ensure it's a valid image (JPEG/PNG) and try again. ({e})"
                
                # You can add more specific error messages if needed, e.g., checking type of 'e'.
                if "invalid image" in str(e).lower() or "cannot identify image file" in str(e).lower():
                    user_message = "The uploaded file is not a valid image or is corrupted. Please upload a JPEG or PNG image."

                return render_template("home.html", message=user_message)
        else:
            # This handles cases where the model failed to load at app startup,
            # preventing prediction attempts.
            return render_template("home.html", message="Error: The AI model could not be loaded. Please check server configuration.")

    # For GET requests (initial page load), render the home.html template with the upload form.
    return render_template("home.html")


# Route to serve the converted README/Project Overview page.
# This page provides comprehensive documentation for the project.
@app.route('/project-overview')
def project_overview():
    # Renders the HTML file generated from the README.md by convert_readme.py.
    # Make sure 'README_interactive_overview.html' exists in your templates folder.
    return render_template('README_interactive_overview.html')


if __name__ == '__main__':
    # Run the Flask application.
    # debug=True: Enables debugging mode (auto-reloads on code changes, provides debugger).
    # host='0.0.0.0': Makes the server accessible from other devices on the local network.
    # port=5000: Specifies the port number the server will listen on.
    app.run(debug=True, host='0.0.0.0', port=5000)
