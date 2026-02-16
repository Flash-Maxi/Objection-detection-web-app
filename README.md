Object Detection Web App
A simple object detection web application built with Python, Flask, and a pre-trained deep learning model.
​
Users can upload images through a web interface and receive the same images back with detected objects highlighted and labeled.
​

Features
Web-based interface for uploading images.
​

Backend in Python using a pre-trained object detection model.
​

Processed images saved and served from a static/results directory.
​

Container-ready using Docker and a provided Dockerfile.
​

Project Structure
text
Objection-detection-web-app/
├── .devcontainer/         # Dev container configuration
├── .vscode/               # VS Code workspace settings
├── static/
│   └── results/           # Output images with detections
├── templates/             # HTML templates (Flask/Jinja2)
├── uploads/               # Uploaded input images
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker image definition
└── .dockerignore          # Files ignored in Docker builds
Prerequisites
Python 3.8+ installed on your system.
​

pip (Python package manager).
​

(Optional) Docker, if you want to run the app in a container.
​

Installation
Clone the repository:

bash
git clone https://github.com/Flash-Maxi/Objection-detection-web-app.git
cd Objection-detection-web-app
Create and activate a virtual environment (recommended):

bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
Install dependencies:

bash
pip install -r requirements.txt
​

Running the App Locally
Make sure you are in the project directory with dependencies installed.
​

Start the Flask app:

bash
python app.py
Open your browser and go to:

text
http://127.0.0.1:5000
Use the UI to upload an image and view the detection results.
​

Docker Usage
You can also build and run the app using Docker.

Build the image:

bash
docker build -t object-detection-web-app .
Run the container:

bash
docker run -p 5000:5000 object-detection-web-app
Open http://localhost:5000 in your browser.
​

Deployment Notes
The current deployment to some hosting platforms may fail because of the relatively large size of the app and its dependencies.
​
Upgrading to a paid/pro plan on the deployment platform or using infrastructure with more generous resource limits (e.g., VPS or dedicated cloud instance) is recommended for production deployment.
​

Technologies Used
Python

Flask (web framework)

HTML/CSS templates for the frontend

Pre-trained object detection model (deep learning)

Docker for containerization

​

Future Improvements
Add support for real-time video or webcam detection.
​

Improve UI/UX and add progress indicators.
​

Log and visualize detection statistics.
​
​

""deployment fails due to the large size of the web app... after purchasing pro version of deployment site, the app will be deploy.""
