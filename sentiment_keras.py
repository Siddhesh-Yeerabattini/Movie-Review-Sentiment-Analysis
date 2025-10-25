# ===========================
# Phase 1: Imports & Setup
# ===========================
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ===========================
# Phase 2: Load IMDB Dataset
# ===========================
# max_features = 10000  # number of distinct words to keep
# maxlen = 300          # cut reviews after 300 words

# print("Loading data...")
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# print(len(X_train), "training samples")
# print(len(X_test), "test samples")

# ===========================
# Phase 2: Load IMDB Dataset from CSV
# ===========================

max_features = 10000  # max number of words in vocab
maxlen = 300          # max words per review

# Load CSV
df = pd.read_csv("data/IMDB Dataset.csv")

# Map labels to 0/1
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split data
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# Tokenize text
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train_text)

X_train = tokenizer.texts_to_sequences(X_train_text)
X_test = tokenizer.texts_to_sequences(X_test_text)

# Pad sequences
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("✅ Data preparation from CSV complete.")


# ===========================
# Phase 3: Preprocess (Padding)
# ===========================
print("Pad sequences (samples x time)...")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("✅ Data preparation complete.")

# ===========================
# Phase 4: Model Building & Training
# ===========================
embedding_dim = 256

model = Sequential([
    Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=maxlen),
    Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

# ===========================
# Phase 5: Training
# ===========================
epochs = 6
batch_size = 64

print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    verbose=1
)

# ===========================
# Phase 6: Evaluation
# ===========================
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print(f"\nTest Loss: {score:.4f}")
print(f"Test Accuracy: {acc:.4f}")


# ===========================
# Phase 7: Save Model
# ===========================
os.makedirs("models", exist_ok=True)

model.save("models/sentiment_lstm.h5")
print("✅ Model saved at models/sentiment_lstm.h5")
