import tensorflow as tf
import pandas as pd
import numpy as np
import json
import datetime
import time
import unicodedata
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense, Masking
from tensorflow.keras.models import Model
from data_loader import NameDataGenerator

# --------------------------
# Config
# --------------------------
MAX_LEN = 30
EMBED_DIM = 32
LSTM_UNITS = 32
BATCH_SIZE = 128
EPOCHS = 5
MIN_WORD_FREQ = 1

# --------------------------
# Helpers
# --------------------------
def normalize_token(token):
    return unicodedata.normalize("NFKC", token).strip()

# --------------------------
# Load Data
# --------------------------
print("Starting data loading and processing...")
start_time = time.time()
sentences, labels = [], []

def process_csv():
    with open("names_dataset.csv", "r", encoding="utf-8") as f:
        next(f)  # skip header
        current_sentence, current_labels = [], []
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    yield current_sentence.copy(), current_labels.copy()
                    current_sentence.clear(), current_labels.clear()
            else:
                parts = line.split(",")
                if len(parts) == 2:
                    token, label = normalize_token(parts[0]), parts[1].strip()
                    if token and label:
                        current_sentence.append(token)
                        current_labels.append(label)
        if current_sentence:
            yield current_sentence, current_labels

max_sequences = 10000000
for i, (sent, lbl) in enumerate(process_csv()):
    sentences.append(sent)
    labels.append(lbl)
    if i + 1 >= max_sequences:
        print(f"Reached maximum sequence limit ({max_sequences})")
        break

print(f"Loaded {len(sentences)} sequences")

# --------------------------
# Encoding
# --------------------------
print("Building vocabulary...")
encoding_start_time = time.time()
word_counter = Counter(token for sent in sentences for token in sent)
tag_counter = Counter(tag for seq in labels for tag in seq)

words = [w for w, c in word_counter.items() if c >= MIN_WORD_FREQ]
tags = list(tag_counter.keys())

word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["PAD"] = 0
word2idx["UNK"] = 1
tag2idx = {t: i for i, t in enumerate(tags)}

json.dump(word2idx, open("word2idx.json", "w"))
json.dump(tag2idx, open("tag2idx.json", "w"))

print(f"Vocabulary size: {len(word2idx)} tokens")
print(f"Number of tag classes: {len(tag2idx)}")

# --------------------------
# Train/Test Split
# --------------------------
print("Performing train-test split...")
split_start_time = time.time()
X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(sentences, labels, test_size=0.05, random_state=42)
print(f"Train set: {len(X_train_data)} sequences, Test set: {len(X_test_data)} sequences")
print(f"Split completed in {time.time() - split_start_time:.2f} seconds")

# --------------------------
# Data Generators
# --------------------------
train_gen = NameDataGenerator(X_train_data, y_train_data, word2idx, tag2idx, max_len=MAX_LEN, batch_size=BATCH_SIZE)
test_gen = NameDataGenerator(X_test_data, y_test_data, word2idx, tag2idx, max_len=MAX_LEN, batch_size=BATCH_SIZE, shuffle=False)

# --------------------------
# Model - Functional API
# --------------------------
print("Building Functional model...")
input_layer = Input(shape=(MAX_LEN,), name="Input")
mask = Masking(mask_value=0)(input_layer)
embed = Embedding(input_dim=len(word2idx), output_dim=EMBED_DIM)(mask)
bilstm = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(embed)
output = TimeDistributed(Dense(len(tag2idx), activation="softmax"))(bilstm)
model = Model(inputs=input_layer, outputs=output, name="functional")

optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# --------------------------
# Training
# --------------------------
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled: using float16 for faster training")
except:
    print("Mixed precision not available, using default precision")

print("Starting training...")
training_start_time = time.time()
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr, tensorboard_callback],
    verbose=1
)
print(f"Training completed in {time.time() - training_start_time:.2f} seconds")

# --------------------------
# Evaluation and Save
# --------------------------
print("Evaluating model...")
eval_start_time = time.time()
eval_results = model.evaluate(test_gen, verbose=1)
print(f"Evaluation completed in {time.time() - eval_start_time:.2f} seconds")

print("Saving Functional model...")
save_start_time = time.time()
model.save("name_entity_functional_model.keras")
print(f"Model saved in {time.time() - save_start_time:.2f} seconds")

print(f"Total processing time: {time.time() - start_time:.2f} seconds")
print(f"Final test accuracy: {eval_results[1]:.4f}")

print("\nTraining summary:")
print(f"- Dataset size: {len(sentences)} sequences")
print(f"- Vocabulary size: {len(word2idx)} tokens")
print(f"- Tag classes: {len(tag2idx)}")
print(f"- Model saved to: name_entity_functional_model.keras")
