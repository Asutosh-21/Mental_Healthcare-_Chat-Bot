mental_health_chatbot/
├── app.py                      # Main Flask application
├── train_model.py              # Script to train the model
├── templates/                  # HTML templates
│   ├── index.html              # Main chat interface
│   └── layout.html             # Base template
├── static/                     # Static files
│   ├── css/
│   │   └── style.css           # Custom styles
│   └── js/
│       └── script.js           # JavaScript for chat interface
├── models/                     # Trained models (created after training)
│   ├── mental_health_chatbot_model.h5
│   ├── words.pkl
│   ├── classes.pkl
│   └── mental_health_intents.json
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── preprocessor.py         # Text preprocessing
└── requirements.txt            # Project dependencies