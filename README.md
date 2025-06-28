# 🔬 HematoVision: Advanced Blood Cell Classification
# https://685b670f38b414e5037b933a--zingy-mooncake-e9ae3d.netlify.app/
# ✨ Project Overview
HematoVision is an innovative AI-powered web application designed for the precise and efficient classification of blood cells. Leveraging advanced deep learning techniques, particularly transfer learning, this project aims to assist pathologists and healthcare professionals in rapid diagnostics and research by automatically identifying different types of blood cells from uploaded images.

This project was developed as part of the Smart Bridge Internship Program, focusing on applying Artificial Intelligence in the healthcare domain.

# 🌟 Key Features
Accurate Classification: Utilizes a fine-tuned CNN for high accuracy in identifying various blood cell types.

Transfer Learning: Employs the MobileNetV2 architecture to ensure efficient training and robust performance.

Intuitive Web Interface: A user-friendly web app built with Flask for easy image uploads and result viewing.

Real-time Feedback: Provides immediate predictions, including class and confidence score, upon submission.

Scalable Approach: The underlying deep learning model is designed to be efficient for practical application.

Open Source Stack: Built entirely on powerful and widely-used open source technologies.

🚀 Technologies Used
Python 3.x: The core programming language for the entire project.

TensorFlow & Keras: Deep Learning framework for model building, training, and prediction.

NumPy: Essential for numerical operations and array manipulation.

Pandas: Used for data handling and manipulation (e.g., creating DataFrames for image paths and labels).

Matplotlib & Seaborn: For data visualization, especially for plotting training history (accuracy, loss) and potentially confusion matrices.

OpenCV (cv2): For image processing tasks (reading, resizing, color conversion).

Flask: A lightweight Python web framework for building the user interface and handling web requests.

Pillow (PIL): Used for image handling and manipulation.

🛠️ Setup Instructions
Follow these steps to get a local copy of the project up and running on your machine.

Prerequisites
Anaconda Navigator: Recommended for managing Python environments and packages. Download from Anaconda Website.

Git: Version control system. Download from Git SCM.

Installation
Clone the Repository (or Download Project Files):
If this were a Git repository, you would clone it:

git clone https://github.com/yourusername/HematoVision.git
cd HematoVision

Since you have the files locally, ensure all project files are organized in a single root folder, e.g., C:\HematoVision.

Initialize Git Repository (If not cloned):
If you've downloaded files manually, navigate to your project folder in Anaconda Prompt and run:

git init
# Configure your Git identity (one-time setup)
git config --global user.email "karavamanojkumarreddy@gmail.com"
git config --global user.name "KARAVA MANOJ KUMAR REDDY"

Create a Dedicated Conda Environment (Recommended):
Open Anaconda Prompt (as Administrator) and create a new environment to avoid dependency conflicts:

conda create -n hemato_env python=3.9  # Or your preferred Python version (e.g., 3.8, 3.10)
conda activate hemato_env

Install Required Python Packages:
With your environment activated, install all necessary libraries. It's best practice to use a requirements.txt file for this.
First, create requirements.txt in your C:\HematoVision folder with these contents:

tensorflow==2.x.x # Use the exact version you trained your model with, e.g., 2.10.0
keras # If you explicitly installed standalone Keras, otherwise tensorflow handles tf.keras
numpy
pandas
scikit-learn
matplotlib
scipy
seaborn
Flask
opencv-python
Pillow # Required for PIL.Image
markdown # For README conversion script

Then, install them:

pip install -r requirements.txt

📂 Project Structure
The project directory (C:\HematoVision) is organized as follows:

# HematoVision/
# app.py   
# trained_model.h5       
# requirements.txt            
# static/(uploaded images will go here)
# templates/(index.html, result.html)             
# dataset/(TRAIN, TEST, etc. folders)
 # model/(Classification Report)
# README.md                      
🚀 How to Run the Application
Ensure all setup steps are completed (prerequisites and installation).

Place your trained model file (trained_model.h5) directly in the C:\HematoVision root directory, next to app.py.

Generate the Interactive Project Overview HTML:
Run the conversion script to create templates/README_interactive_overview.html:

python convert_readme.py

Open Anaconda Prompt (as Administrator).

Navigate to your project's root directory:

cd C:\HematoVision

Activate your Conda environment:

conda activate hemato_env # Or whatever you named your environment

Start the Flask application:

python app.py

You should see output similar to this, indicating the server is running and the model loaded:

✅ Model loaded successfully from C:\HematoVision\trained_model.h5
* Serving Flask app 'app'
* Debug mode: on
* Running on all addresses (0.0.0.0)
* Running on http://127.0.0.1:5000
* Running on http://YOUR_LOCAL_IP:5000
Press CTRL+C to quit

💻 Usage
Once the Flask application is running, open your web browser:

Blood Cell Classification Tool:

Go to http://127.0.0.1:5000/ (for access from your local machine).

If running on 0.0.0.0, others on your same local network can access it using your computer's local IP address (e.g., http://192.168.1.100:5000/).

On the page, click "Choose File", select a blood cell image (JPEG or PNG), and click "Classify Image" to see the prediction.

Interactive Project Overview (README):

Go to http://127.0.0.1:5000/project-overview in your browser.

This page provides a comprehensive, interactive overview of the project's details, features, and technical aspects.

🔮 Future Enhancements
Advanced UI/UX: Implement more dynamic and visually engaging elements using JavaScript frameworks, potentially integrating a progress bar for uploads/predictions.

Model Management: Develop features for easily updating the model or comparing different model versions.

Batch Prediction: Allow users to upload multiple images for classification at once.

Detailed Reports: Generate comprehensive reports for classification results, including probabilities for all classes.

Containerization: Package the application using Docker for easier deployment and environment consistency.

Cloud Deployment: Host the application on a public cloud platform (e.g., Heroku, AWS, GCP) for public accessibility and scalability.

🤝 Contributing
We welcome contributions to the HematoVision project! If you have suggestions or improvements, please feel free to open an issue or submit a pull request (if this were a Git repo).

📄 License
This project is licensed under the MIT License - see the LICENSE file for details (if you create one).

📧 Contact
# Project Lead: Karava Manoj Kumar Reddy
# Team Members: Kolla Manikanta, Kulagiri Indu Priya and Karuturi Mahendranadh Chowdhury
# Email: karavamanojkumarreddy@gmail.com
Smart Bridge Internship: [https://apsche.smartinternz.com/Student/guided_project_workspace/42645]

Developed as part of the Smart Bridge AI Internship Program.
