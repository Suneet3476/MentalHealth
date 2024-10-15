from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import torch
import nltk

app = Flask(__name__)

# Download vader_lexicon
nltk.download('vader_lexicon')

# Load Hugging Face model and tokenizer (DialoGPT-medium) only when needed
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# Initialize VADER sentiment analysis
sentiment_analyzer = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment_score = sentiment_analyzer.polarity_scores(text)
    compound_score = sentiment_score['compound']
    
    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    else:
        return "neutral"

# Function to generate a response using Hugging Face
def generate_response(user_input):
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").half()
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
    response_ids = model.generate(inputs, max_length=50, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    
    del model
    torch.cuda.empty_cache()
    
    return response_text

# Function to suggest calming music
def suggest_music(sentiment):
    if sentiment == "negative":
        return "How about some calming music? Try listening to 'Weightless' by Marconi Union or 'Clair de Lune' by Debussy."
    elif sentiment == "positive":
        return "You're in a good mood! How about some upbeat tunes? Try 'Happy' by Pharrell Williams!"
    else:
        return "Would you like to listen to some relaxing music? You could try 'Lo-fi beats' to chill."

# Route to handle user input and AI response
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('user_input')
    
    sentiment = analyze_sentiment(user_input)
    ai_response = generate_response(user_input)
    music_suggestion = suggest_music(sentiment)
    
    return jsonify({
        "user_input": user_input,
        "ai_response": ai_response,
        "sentiment": sentiment,
        "music_suggestion": music_suggestion
    })

# Serve the HTML interface
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
