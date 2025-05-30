# Mental_Healthcare-_Chat-Bot
This project will help beginners understand the full stack development process while creating a meaningful application for mental health support.
Mental Health Chatbot - Flask Application
A simple mental health support chatbot built with Flask, TensorFlow, and NLTK. This project is for educational purposes only and not a substitute for professional mental health care.
Features

Intent-based chatbot responses
Simple and clean chat interface
Crisis detection and appropriate responses
Optional advanced responses using transformer models

Setup Instructions
1. Prerequisites

Python 3.8 or higher
pip (Python package installer)
Virtual environment (recommended)

2. Clone the repository
bashgit clone <your-repository-url>
cd mental_health_chatbot
3. Create and activate a virtual environment
bash# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
4. Install dependencies
bashpip install -r requirements.txt
5. Download NLTK data
python# Run Python in interactive mode
python

# In Python shell
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
exit()
6. Train the chatbot model
Before running the Flask app, you need to train the model:
bashpython train_model.py
This will create the necessary model files in the models directory.
7. Run the Flask application
bashpython app.py


app.py: Main Flask application
templates/: HTML templates for the web interface
static/: CSS and JavaScript files
models/: Trained machine learning models
utils/: Utility functions for preprocessing and prediction
train_model.py: Script to train the chatbot model
