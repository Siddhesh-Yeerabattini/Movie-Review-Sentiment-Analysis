from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

max_features = 10000
maxlen = 300

# Load model and word index
model = load_model("models/sentiment_lstm.h5")
word_index = imdb.get_word_index()

def encode_review(text):
    tokens = text.lower().split()
    encoded = [word_index.get(word, 2)+3 for word in tokens]  # 2=UNK
    return sequence.pad_sequences([encoded], maxlen=maxlen)

def predict_sentiment(text):
    encoded = encode_review(text)
    score = model.predict(encoded)[0][0]
    return "Positive" if score > 0.5 else "Negative", score

# Example usage
text = "The movie was painfully slow and predictable. Halfway through, I was already bored."
sentiment, score = predict_sentiment(text)
print(f"Sentiment: {sentiment}, Score: {score:.4f}")
