import numpy as np
import pandas as pd
import re
import json
import pickle
import random
import nltk
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Ensure directories exist
os.makedirs('models', exist_ok=True)

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

print("Creating mental health intents dataset...")

# Simple mental health conversation dataset
# Enhanced mental health conversation dataset
mental_health_intents = {
    "intents": [
        # Original intents
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up", "Good day", "Good morning", "Good evening"],
            "responses": ["Hello! I'm your mental health assistant. How can I help you today?", 
                         "Hi there! I'm here to listen and offer support. How are you feeling?", 
                         "Hello! I'm here to chat about anything that's on your mind.",
                         "Welcome! I'm your wellness companion. How can I support you today?"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye", "I'm leaving", "Have a good day", "Talk later", "Until next time"],
            "responses": ["Take care of yourself!", 
                         "If you need to talk again, I'll be here. Goodbye!", 
                         "Remember to be kind to yourself. Bye!",
                         "Wishing you well until our next conversation. Take care!"]
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you", "That's helpful", "Thank you so much", "I appreciate it", "That helps"],
            "responses": ["Happy to help!", 
                         "Any time!", 
                         "My pleasure, I'm here whenever you need someone to talk to.",
                         "I'm glad I could be of assistance. Remember, supporting your mental health is an ongoing journey."]
        },
        {
            "tag": "anxiety",
            "patterns": ["I'm feeling anxious", "I have anxiety", "I'm worried all the time", "I can't stop worrying", "My anxiety is bad today", 
                        "Panic attack", "My mind won't stop racing", "I feel on edge"],
            "responses": ["I understand anxiety can be overwhelming. Deep breathing exercises might help in the moment.", 
                         "Anxiety is tough to deal with. Have you tried any relaxation techniques?", 
                         "It's okay to feel anxious sometimes. Would you like to talk about what's causing your anxiety?",
                         "For immediate anxiety relief, try the 5-4-3-2-1 grounding technique: acknowledge 5 things you see, 4 things you can touch, 3 things you hear, 2 things you smell, and 1 thing you taste.",
                         "When anxiety strikes, progressive muscle relaxation can help - tense and then release each muscle group in your body, starting from your toes and working upward."]
        },
        {
            "tag": "depression",
            "patterns": ["I'm feeling depressed", "I'm so sad", "I have depression", "I feel hopeless", "I don't see the point anymore", 
                        "Everything feels empty", "I can't feel happy", "Nothing interests me anymore"],
            "responses": ["I'm sorry you're feeling this way. Depression can make everything seem harder.", 
                         "Remember that you're not alone in these feelings. Have you spoken to anyone else about how you're feeling?", 
                         "Depression is a real challenge. Small self-care activities might help a little bit right now.",
                         "Depression can drain your energy. Setting tiny achievable goals each day - even just getting out of bed or taking a shower - can be meaningful victories.",
                         "While I'm here to listen, depression often responds well to professional treatment. Consider reaching out to a healthcare provider who can discuss therapy and other options."]
        },
        {
            "tag": "stress",
            "patterns": ["I'm stressed out", "There's too much pressure", "I can't handle the stress", "Everything is stressful", 
                        "I'm overwhelmed", "Too much to do", "Burnout", "I'm at my limit"],
            "responses": ["Stress can be really overwhelming. Taking short breaks might help.", 
                         "When we're under stress, our thinking gets affected too. What's one small thing you could do for yourself right now?", 
                         "Dealing with stress is tough. Have you tried any stress management techniques?",
                         "Chronic stress can affect your whole body. Regular relaxation practices like deep breathing or gentle movement can help regulate your nervous system.",
                         "Sometimes writing down your stressors and categorizing them into 'things I can control' and 'things I can't control' helps create a clearer perspective."]
        },
        {
            "tag": "sleep",
            "patterns": ["I can't sleep", "I have insomnia", "I'm having trouble sleeping", "How can I sleep better", 
                        "I wake up a lot", "Bad dreams", "Sleep hygiene", "Tired but can't sleep"],
            "responses": ["Sleep problems are common with mental health challenges. Try to maintain a regular sleep schedule if possible.", 
                         "Have you tried creating a relaxing bedtime routine?", 
                         "Screen time before bed can make sleep harder. Reading or gentle stretching might help instead.",
                         "Sleep hygiene practices that can help include keeping your bedroom cool and dark, avoiding caffeine after noon, and using your bed only for sleep.",
                         "If racing thoughts keep you awake, try keeping a 'worry journal' next to your bed to write down thoughts that can be addressed tomorrow."]
        },
        {
            "tag": "self_care",
            "patterns": ["How do I take care of myself", "What is self care", "Self care tips", "How to feel better", 
                        "Self care practices", "Ways to improve mood", "How to be kinder to myself"],
            "responses": ["Self-care includes basic needs like rest, nutrition, and exercise, plus activities that bring you joy.", 
                         "Even small acts of self-care can help. Could you drink a glass of water or step outside for fresh air?", 
                         "Self-care looks different for everyone. What's one small thing that usually helps you feel a bit better?",
                         "Self-care isn't selfish - it's necessary. Think of it as maintaining your personal battery so you have energy for yourself and others.",
                         "An effective self-care routine addresses physical, emotional, social, and spiritual needs. Which area feels most neglected for you right now?"]
        },
        {
            "tag": "professional_help",
            "patterns": ["I need a therapist", "How do I get help", "I need professional help", "Should I see a doctor", "Find me a counselor", 
                        "How to find therapy", "Mental health services", "Psychiatrist vs therapist"],
            "responses": ["I recommend speaking with a mental health professional who can provide proper care. Many offer telehealth options now.", 
                         "A therapist or counselor could provide the support you need. Would you like information on finding professional help?", 
                         "This is an important step. Speaking with your doctor or a mental health professional is a good place to start.",
                         "To find a therapist, you can check with your insurance provider, use online directories like Psychology Today, or ask your primary care doctor for referrals.",
                         "Different types of mental health professionals offer different services. Psychiatrists can prescribe medication, while psychologists, counselors, and social workers typically provide talk therapy."]
        },
        {
            "tag": "crisis",
            "patterns": ["I want to hurt myself", "I want to die", "I'm thinking of suicide", "I don't want to live", "I want to end it all", 
                        "No reason to live", "Plan to kill myself", "Self harm", "Suicidal"],
            "responses": ["I'm concerned about what you're sharing. Please reach out to a crisis helpline immediately - they are available 24/7 at 988 in the US. They're there to help.", 
                         "This sounds serious, and you deserve immediate support. Please call a crisis line like 988 (US) or go to your nearest emergency room.", 
                         "You're not alone, and help is available. Please call 988 or your local emergency number right away to speak with someone who can help.",
                         "These feelings are incredibly difficult, but they can change with proper support. Please contact emergency services or go to the nearest emergency room.",
                         "Your life matters. Please reach out to the National Suicide Prevention Lifeline at 988 right away - they have trained counselors available 24/7."]
        },
        {
            "tag": "motivation",
            "patterns": ["I have no motivation", "I can't get anything done", "Everything feels pointless", "How do I get motivated", 
                        "Can't focus", "No energy to do things", "Too tired to try", "How to start tasks"],
            "responses": ["Lack of motivation can be related to mental health. Starting with tiny tasks might help build momentum.", 
                         "It's okay to take things one small step at a time. What's the smallest possible thing you could do?", 
                         "Be gentle with yourself during periods of low motivation. Your worth isn't measured by productivity.",
                         "Try the 'five-minute rule' - commit to just five minutes of an activity. Often, getting started is the hardest part.",
                         "Motivation sometimes follows action rather than precedes it. Small accomplishments can help rebuild your sense of capability."]
        },
        {
            "tag": "mindfulness",
            "patterns": ["What is mindfulness", "How do I practice mindfulness", "Mindfulness techniques", "How to be present", 
                        "Meditation tips", "Stay in the moment", "Mindfulness exercises", "Mindful breathing"],
            "responses": ["Mindfulness means paying attention to the present moment without judgment. Simply noticing your breath is a start.", 
                         "You can practice mindfulness through meditation, mindful walking, or just paying full attention to everyday activities.", 
                         "Mindfulness can help us observe our thoughts and feelings without being overwhelmed by them.",
                         "A simple mindfulness practice is to focus completely on a routine activity, like brushing your teeth or washing dishes, using all your senses.",
                         "The STOP practice can be helpful: Stop what you're doing, Take a breath, Observe what's happening in your body and mind, then Proceed with awareness."]
        },
        {
            "tag": "unsure",
            "patterns": ["I don't know what to do", "I'm confused", "I'm not sure", "What should I do", 
                        "I'm lost", "Need direction", "Can't decide", "Feel stuck"],
            "responses": ["It's okay not to have all the answers. Would it help to talk through what you're experiencing?", 
                         "Uncertainty is challenging. Perhaps we could break things down into smaller pieces?", 
                         "Sometimes when we're overwhelmed, it helps to focus just on the next small step.",
                         "Not knowing what to do is a common experience. Could we explore what options feel most aligned with your values?",
                         "When feeling uncertain, sometimes it helps to ask: 'What would I tell a friend in this situation?'"]
        },
        
        # New intents focused on yoga and mindful movement
        {
            "tag": "yoga",
            "patterns": ["Tell me about yoga for mental health", "Yoga for anxiety", "Yoga poses for stress", "How does yoga help depression", 
                        "Beginner yoga for mental health", "Yoga breathing techniques", "Yoga benefits for mind"],
            "responses": ["Yoga combines physical postures, breathing exercises, and meditation, which can help regulate your nervous system and reduce stress hormones.", 
                         "For anxiety, gentle forward folds and child's pose can be calming. Slow, controlled breathing during these poses helps activate your relaxation response.", 
                         "Even 5-10 minutes of gentle yoga can help shift your mental state. Cat-cow pose, gentle twists, and mountain pose are accessible starting points.",
                         "Yoga isn't just about flexibility - it teaches mindfulness through body awareness, which can help interrupt patterns of negative thinking.",
                         "If traditional yoga feels intimidating, chair yoga is an accessible option that offers many of the same mental health benefits."]
        },
        {
            "tag": "meditation",
            "patterns": ["How to meditate", "Meditation for beginners", "Meditation techniques", "Guided meditation", 
                        "Meditation benefits", "How long should I meditate", "Different types of meditation"],
            "responses": ["Meditation can be as simple as sitting quietly and focusing on your breath for a few minutes. When your mind wanders, gently bring it back to your breath.", 
                         "For beginners, guided meditations can be helpful. Many apps and YouTube channels offer free options ranging from 5-20 minutes.", 
                         "Different types of meditation include focused attention (on breath, a word, or object), loving-kindness (generating feelings of goodwill), and body scan (systematically relaxing the body).",
                         "Studies show even 5-10 minutes of daily meditation can improve attention, reduce stress, and support emotional regulation over time.",
                         "If sitting meditation is difficult, try walking meditation, where you focus on the sensations of each step, or eating meditation, where you fully engage with the experience of eating."]
        },
        {
            "tag": "exercise",
            "patterns": ["Exercise for mental health", "Working out for depression", "Best exercise for anxiety", "Physical activity mood", 
                        "How does exercise help mental health", "Exercise motivation", "Simple workouts for mental health"],
            "responses": ["Exercise releases endorphins and other feel-good chemicals that can improve mood. Even a 10-minute walk can make a difference.", 
                         "For depression, activities that get your heart rate up like walking, jogging, or dancing can be particularly beneficial - aim for 30 minutes most days if possible.", 
                         "With anxiety, gentler forms of movement like walking, swimming, or cycling may be better than high-intensity exercise, which can sometimes temporarily increase anxiety symptoms.",
                         "The best exercise for mental health is one you'll actually do consistently. What activities have you enjoyed in the past?",
                         "If motivation is an issue, try 'habit stacking' by adding brief movement to existing routines - like squats while brushing teeth or calf raises while waiting for coffee."]
        },
        {
            "tag": "breathing",
            "patterns": ["Breathing exercises", "Deep breathing for anxiety", "Breathing techniques", "Box breathing", 
                        "Calming breath", "Breath work", "How to breathe when anxious", "4-7-8 breathing"],
            "responses": ["Deep breathing activates your parasympathetic nervous system, which helps calm your stress response. Try breathing in for 4 counts, hold for 1, out for 6.", 
                         "Box breathing is helpful for anxiety: breathe in for 4 counts, hold for 4, exhale for 4, hold for 4, and repeat.", 
                         "The 4-7-8 technique involves breathing in for 4, holding for 7, and exhaling for 8. This extended exhale can be especially calming.",
                         "Even just taking 3 conscious breaths - breathing slowly into your belly rather than your chest - can help reset your nervous system in stressful moments.",
                         "For quick stress relief, try the physiological sigh: two inhales through your nose followed by a long exhale through your mouth. This helps release carbon dioxide that builds up during stress."]
        },
        {
            "tag": "bipolar",
            "patterns": ["I think I'm bipolar", "Bipolar symptoms", "Mood swings", "Manic depression", 
                        "Bipolar disorder", "Mania and depression", "Bipolar treatment", "Managing bipolar"],
            "responses": ["Bipolar disorder involves distinct periods of depression and mania or hypomania. A proper diagnosis from a psychiatrist is important, as treatment differs from depression alone.", 
                         "Managing bipolar disorder often includes medication, therapy, regular sleep patterns, stress management, and avoiding substances that can trigger episodes.", 
                         "If you're experiencing extreme mood shifts between depression and periods of high energy, decreased need for sleep, and increased impulsivity, it's important to discuss this with a healthcare provider.",
                         "Tracking your moods, sleep, medication, and potential triggers can help you and your healthcare provider manage bipolar symptoms more effectively.",
                         "Having a bipolar disorder diagnosis doesn't define you, and many people live fulfilling lives with proper treatment and support."]
        },
        {
            "tag": "ocd",
            "patterns": ["I think I have OCD", "Obsessive thoughts", "Compulsive behavior", "OCD symptoms", 
                        "Intrusive thoughts", "OCD treatment", "Pure O OCD", "Fear of contamination"],
            "responses": ["OCD involves unwanted thoughts, images, or urges (obsessions) and repetitive behaviors or mental acts (compulsions) aimed at reducing distress. It's more than just liking things organized.", 
                         "Effective treatments for OCD include Exposure and Response Prevention (ERP) therapy and sometimes medication. A mental health professional can help determine the best approach.", 
                         "Many people with OCD experience shame about their thoughts, but having intrusive thoughts doesn't reflect who you are as a person.",
                         "OCD themes vary widely - from contamination fears to harm thoughts to need for symmetry - but the pattern of obsession and compulsion is consistent across subtypes.",
                         "If intrusive thoughts and ritualistic behaviors are interfering with your daily life, it's important to seek help from someone specializing in OCD treatment."]
        },
        {
            "tag": "ptsd",
            "patterns": ["I think I have PTSD", "Trauma symptoms", "Flashbacks", "PTSD treatment", 
                        "Trauma therapy", "Nightmares about trauma", "EMDR therapy", "Complex PTSD"],
            "responses": ["PTSD can develop after experiencing or witnessing traumatic events. Symptoms include flashbacks, nightmares, severe anxiety, and uncontrollable thoughts about the event.", 
                         "There are effective treatments for PTSD, including trauma-focused cognitive behavioral therapy, EMDR, and sometimes medication.", 
                         "Trauma responses are your brain's normal reaction to abnormal events. It's not a weakness or character flaw.",
                         "Grounding techniques can help during flashbacks or triggered moments. Focus on what you can see, touch, hear, smell, and taste in your present environment.",
                         "Complex PTSD can develop from prolonged trauma, especially in childhood. Its symptoms include difficulty regulating emotions and maintaining relationships, in addition to traditional PTSD symptoms."]
        },
        {
            "tag": "adhd",
            "patterns": ["I think I have ADHD", "Trouble focusing", "Can't concentrate", "Always distracted", 
                        "Executive dysfunction", "Adult ADHD", "ADHD symptoms", "ADHD coping strategies"],
            "responses": ["ADHD involves persistent patterns of inattention, hyperactivity-impulsivity, or both. In adults, it often presents as difficulty with organization, time management, and completing tasks.", 
                         "If you suspect you have ADHD, consider seeking evaluation from a healthcare provider. Proper diagnosis can open doors to helpful treatments and accommodations.", 
                         "Many people with ADHD find strategies like breaking tasks into smaller steps, using timers, and creating external structure helpful for managing symptoms.",
                         "ADHD often comes with strengths too, like creativity, hyperfocus on interesting topics, and thinking outside the box.",
                         "Body-doubling (working alongside someone else) and accountability partners can help with task initiation and completion if you have ADHD-related executive function challenges."]
        },
        {
            "tag": "eating_disorders",
            "patterns": ["Eating disorder help", "Anorexia", "Bulimia", "Binge eating", "Body image issues", 
                        "Food anxiety", "Orthorexia", "Disordered eating", "Afraid of gaining weight"],
            "responses": ["Eating disorders are serious mental health conditions that require professional support. They're about much more than food - they often involve complex emotional needs and coping mechanisms.", 
                         "Recovery from eating disorders is possible. Treatment typically includes therapy, nutritional counseling, and sometimes medication.", 
                         "If you're struggling with food, eating, or body image in ways that impact your physical or mental health, please consider reaching out to a healthcare provider specialized in eating disorders.",
                         "Eating disorders affect people of all genders, ages, body sizes, and backgrounds. You deserve support regardless of your appearance or weight.",
                         "The National Eating Disorders Association (NEDA) helpline can provide resources and support: 1-800-931-2237 or text 'NEDA' to 741741."]
        },
        {
            "tag": "nutrition",
            "patterns": ["Diet and mental health", "Foods for depression", "Nutrition anxiety", "What to eat for mood", 
                        "Best foods for mental health", "Mental health diet", "Sugar and anxiety", "Anti-inflammatory diet"],
            "responses": ["Nutrition can influence mental health. Foods rich in omega-3s (like fatty fish), antioxidants (colorful fruits and vegetables), and fiber may support brain health.", 
                         "Regular meals help stabilize blood sugar, which can prevent mood swings. Try not to go too long without eating.", 
                         "The gut-brain connection is powerful. Fermented foods and fiber help support gut bacteria that produce neurotransmitters affecting mood.",
                         "While no specific diet is proven to cure mental health conditions, a Mediterranean-style eating pattern has been associated with lower depression risk in studies.",
                         "Hydration affects brain function and mood. Even mild dehydration can worsen concentration and increase fatigue."]
        },
        {
            "tag": "social_anxiety",
            "patterns": ["Social anxiety", "Afraid of people", "Fear of judgment", "Nervous in social situations", 
                        "Can't talk to people", "Social phobia", "Performance anxiety", "Fear of embarrassment"],
            "responses": ["Social anxiety involves intense fear of social situations and worry about being judged, embarrassed, or humiliated. It's more than just shyness.", 
                         "Cognitive behavioral therapy (CBT) is particularly effective for social anxiety. It helps challenge negative thoughts and gradually face feared situations.", 
                         "Small steps like making brief eye contact or asking a store clerk one question can help build confidence over time.",
                         "Remember that most people are focused on themselves, not scrutinizing you. And those who matter won't judge you harshly for normal human moments.",
                         "Preparation can help ease social anxiety. Practice what you might say, arrive early to get comfortable with the environment, or bring a supportive friend to challenging events."]
        },
        {
            "tag": "grief",
            "patterns": ["Dealing with grief", "Lost someone", "Bereavement", "Coping with death", "Stages of grief", 
                        "Grief process", "Missing someone who died", "Anniversary of death"],
            "responses": ["Grief is a natural response to loss, and there's no single 'right' way to grieve. Your experience is valid however it unfolds.", 
                         "The idea of grief moving through neat stages is a myth. Most people experience it as a winding path with good days and hard days.", 
                         "Allow yourself to feel whatever emotions arise without judgment. Sadness, anger, relief, and even moments of joy can all be part of grieving.",
                         "Grief doesn't have a timeline. Be patient with yourself and resist pressure to 'move on' or 'get over it.'",
                         "Finding ways to honor and remember your loved one can be healing - whether through rituals, sharing stories, or creating something in their memory."]
        },
        {
            "tag": "loneliness",
            "patterns": ["I'm lonely", "No friends", "Feel isolated", "How to make friends", "Always alone", 
                        "Social isolation", "Lonely but can't connect", "Need human connection"],
            "responses": ["Loneliness is a common human experience, but that doesn't make it any less painful. You're not alone in feeling alone.", 
                         "Small steps toward connection can help - joining a class, volunteering, or using apps designed for friendship can create opportunities to meet people.", 
                         "Quality matters more than quantity with relationships. Even one or two meaningful connections can significantly reduce loneliness.",
                         "Online communities centered around shared interests can be a good starting point, especially if social situations are challenging.",
                         "Being kind to yourself during lonely periods is important. Treat yourself with the same compassion you'd offer a friend in your situation."]
        },
        {
            "tag": "work_stress",
            "patterns": ["Work stress", "Job burnout", "Hate my job", "Workplace anxiety", "Career stress", 
                        "Toxic workplace", "Work-life balance", "Dealing with coworkers"],
            "responses": ["Work stress is common but shouldn't consume your life. Setting boundaries between work and personal time is essential for wellbeing.", 
                         "Signs of burnout include exhaustion, cynicism, and reduced efficacy. If you're experiencing these, prioritizing rest and recovery is important.", 
                         "Small breaks throughout the workday can help - even 5 minutes of stepping outside or deep breathing can reset your stress response.",
                         "Consider what aspects of work you can control vs. what you can't. Focus your energy on improving what's within your control.",
                         "If your workplace is significantly impacting your mental health, it may be worth exploring other options - whether that's a different role, company, or setting boundaries."]
        },
        {
            "tag": "relationship_issues",
            "patterns": ["Relationship problems", "Partner issues", "Marriage troubles", "How to communicate better", 
                        "Fighting with spouse", "Trust issues", "Relationship anxiety", "Attachment issues"],
            "responses": ["Relationship challenges are normal, but how we handle them makes a difference. Communication skills like using 'I' statements and active listening can help.", 
                         "During conflicts, taking a short break when emotions run high can prevent saying things you might regret.", 
                         "Healthy relationships include mutual respect, trust, good communication, and support for each other's independence.",
                         "Relationship patterns often stem from early attachment experiences. Understanding your attachment style can help improve how you relate to others.",
                         "If relationship problems are persistent or severe, couples counseling can provide tools and a safe space to work through challenges together."]
        },
        {
            "tag": "panic_attacks",
            "patterns": ["Having panic attacks", "Panic disorder", "Feel like I'm dying", "Heart racing anxiety", 
                        "Can't breathe panic", "Sudden intense fear", "Panic attack help", "Panic symptoms"],
            "responses": ["Panic attacks involve intense fear with physical symptoms like racing heart, shortness of breath, and dizziness. Though terrifying, they're not dangerous and typically peak within 10 minutes.", 
                         "During a panic attack, focus on slow, deep breathing - in through nose for 4 counts, out through mouth for 6. Remind yourself it will pass.", 
                         "Grounding techniques can help during panic: name 5 things you see, 4 things you can touch, 3 things you hear, 2 things you smell, and 1 thing you taste.",
                         "Cognitive behavioral therapy is very effective for panic disorder, helping you reinterpret physical sensations and gradually face feared situations.",
                         "If panic attacks are recurring, avoiding caffeine, alcohol, and irregular meal times may help reduce their frequency."]
        },
        {
            "tag": "seasonal_depression",
            "patterns": ["Seasonal depression", "Winter blues", "SAD", "Seasonal affective disorder", 
                        "Depressed in winter", "Light therapy", "Mood changes with seasons", "Dark weather mood"],
            "responses": ["Seasonal Affective Disorder (SAD) involves depression symptoms that occur at specific times of year, most commonly during fall and winter months.", 
                         "Light therapy using a special bright light box for 20-30 minutes each morning can be effective for many people with SAD.", 
                         "Maintaining vitamin D levels through diet, supplements (with healthcare provider guidance), or brief midday sun exposure when possible may help manage seasonal symptoms.",
                         "Regular physical activity, especially outdoors in natural daylight, can significantly improve mood during darker months.",
                         "Planning enjoyable activities and social connections during your difficult season can provide points of light during challenging times."]
        },
        {
            "tag": "substance_use",
            "patterns": ["Drinking too much", "Drug problem", "Substance abuse", "Addiction help", 
                        "Can't stop using", "Alcohol dependence", "Recovery resources", "Sober curious"],
            "responses": ["Substance use exists on a spectrum, and recognizing when it's becoming problematic is an important first step.", 
                         "Support is available through healthcare providers, therapists specializing in addiction, and free groups like AA, NA, SMART Recovery, and Refuge Recovery.", 
                         "Many find the HALT check helpful - asking if you're Hungry, Angry, Lonely, or Tired before using substances, as these states can trigger cravings.",
                         "Recovery looks different for everyone. Some benefit from abstinence, others from harm reduction approaches. What matters is finding what works for your health and goals.",
                         "If you're physically dependent on alcohol or certain drugs, medical supervision during withdrawal is important for safety. Please consult a healthcare provider."]
        },
        {
            "tag": "positive_psychology",
            "patterns": ["How to be happier", "Positive psychology", "Gratitude practice", "Finding joy", 
                        "Building resilience", "Happiness habits", "Flourishing", "Well-being practices"],
            "responses": ["Positive psychology focuses on strengths and well-being rather than just treating problems. Practices like gratitude journaling and using your core strengths can increase life satisfaction.", 
                         "Regular gratitude practice - noting 3 specific things you're grateful for daily - has been shown to improve mood and outlook over time.", 
                         "Strong social connections are consistently linked to happiness and longevity. Investing time in meaningful relationships is one of the most effective ways to increase well-being.",
                         "Finding a balance between pursuing meaningful goals and practicing present-moment awareness contributes to sustainable happiness.",
                         "Happiness isn't about feeling positive all the time, but rather having the full range of emotions while maintaining an underlying sense of meaning and resilience."]
        }
    ]
}
# Save intents to a file
with open('models/mental_health_intents.json', 'w') as f:
    json.dump(mental_health_intents, f, indent=4)

print("Created and saved mental health intents.")

# --- Preprocess the data ---
print("Preprocessing data...")

# Lists to store data
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process each intent
for intent in mental_health_intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add to documents
        documents.append((word_list, intent['tag']))
        # Add to classes if not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and filter words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save processed data
pickle.dump(words, open('models/words.pkl', 'wb'))
pickle.dump(classes, open('models/classes.pkl', 'wb'))

print("Preprocessed data and saved words and classes.")

# --- Create training data ---
print("Creating training data...")

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Create bag of words for each pattern
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    for word in words:
        bag.append(1 if word in word_patterns else 0)
    
    # Create output row (0 for each tag and 1 for current tag)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    training.append([bag, output_row])

# Shuffle and convert to numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Split into features and labels
train_x = list(training[:, 0])
train_y = list(training[:, 1])

print("Training data created.")

# --- Build and train the model ---
print("Building and training the model...")

# Create model - a simple neural network
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
history = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('models/mental_health_chatbot_model.h5')

print("Model training complete! The model has been saved to 'models/mental_health_chatbot_model.h5'")
print("You can now run 'app.py' to start the Flask application.")