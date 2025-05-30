from flask import Flask, render_template, request, jsonify
import os
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocessor import clean_up_sentence, bag_of_words
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize Flask app
app = Flask(__name__)

# Load files and models
try:
    # Basic intent model
    words = pickle.load(open('models/words.pkl', 'rb'))
    classes = pickle.load(open('models/classes.pkl', 'rb'))
    model = load_model('models/mental_health_chatbot_model.h5')
    
    # Load intents
    with open('models/mental_health_intents.json', 'r') as f:
        intents = json.load(f)
    
    # Safety responses
    safety_responses = {
        "crisis": "I notice you might be in distress. Please remember that immediate help is available by calling 988 (US) or your local crisis line. Would you like information about mental health resources?",
        "harm": "I'm concerned about what you're sharing. Please reach out to a crisis helpline immediately at 988 (US) or your local emergency number.",
        "emergency": "This sounds urgent. Please contact emergency services or go to your nearest emergency room. Your wellbeing is important."
    }
    
    print("Models loaded successfully!")
    
    # Advanced transformer model (uncomment if you want to use it)
    # model_name = "google/flan-t5-small"
    # tokenizer = AutoTokenizer.from_pretrained("models/chatbot_tokenizer")
    # model_advanced = AutoModelForSeq2SeqLM.from_pretrained("models/chatbot_llm_model")
    # text_generator = pipeline("text2text-generation", model=model_advanced, tokenizer=tokenizer)
    # print("Transformer model loaded successfully!")

except Exception as e:
    print(f"Error loading models: {e}")

# Helper functions
def predict_class(sentence):
    """
    Predict the class (intent) of the sentence
    """
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    
    # Set a threshold for prediction confidence
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    
    return return_list

def get_response(intents_list, intents_json):
    """
    Get a response based on the predicted intent
    """
    # If no intent was predicted with confidence
    if not intents_list:
        return "I'm not sure I understand. Could you rephrase that?"
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            # Get a random response from the intent
            import random
            result = random.choice(i['responses'])
            
            # Special handling for crisis messages
            if tag == "crisis":
                return safety_responses["crisis"]
            
            return result
    
    return "I'm not sure how to respond to that."

# Check for crisis keywords in the message
def check_for_crisis(message):
    crisis_keywords = ["suicide", "kill myself", "end my life", "don't want to live", 
                      "hurt myself", "self harm", "die", "death"]
    
    # Convert to lowercase for case-insensitive matching
    message_lower = message.lower()
    
    for keyword in crisis_keywords:
        if keyword in message_lower:
            return True
    return False

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_message = request.json['message']
    
    # Check for crisis indicators first
    if check_for_crisis(user_message):
        return jsonify({'response': safety_responses["crisis"]})
    
    # Get regular chatbot response
    ints = predict_class(user_message)
    response = get_response(ints, intents)
    
    return jsonify({'response': response})

# Advanced transformer-based response (uncomment if using)
# @app.route('/get_advanced_response', methods=['POST'])
# def get_advanced_response():
#     user_message = request.json['message']
#     
#     # Check for crisis indicators first
#     if check_for_crisis(user_message):
#         return jsonify({'response': safety_responses["crisis"]})
#     
#     # Generate response using transformer model
#     prompt = f"As a supportive mental health chatbot, respond to: {user_message}"
#     response = text_generator(prompt, max_length=100, num_return_sequences=1)
#     
#     return jsonify({'response': response[0]['generated_text']})

if __name__ == '__main__':
    app.run(debug=True)