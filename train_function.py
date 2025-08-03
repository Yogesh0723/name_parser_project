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

# --------------------------
# Config
# --------------------------
MAX_LEN = 30
EMBED_DIM = 32
LSTM_UNITS = 32
BATCH_SIZE = 256
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

max_sequences = 3000000
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
# Build vocabulary from all data
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

# Process data in batches to reduce memory usage
print("Encoding sequences in batches...")
BATCH_SIZE_ENCODING = 300000
num_batches = (len(sentences) + BATCH_SIZE_ENCODING - 1) // BATCH_SIZE_ENCODING

X, y = [], []
for batch_idx in range(num_batches):
    start_idx = batch_idx * BATCH_SIZE_ENCODING
    end_idx = min(start_idx + BATCH_SIZE_ENCODING, len(sentences))
    
    # Encode this batch
    X_batch = [[word2idx.get(w, 1) for w in s] for s in sentences[start_idx:end_idx]]
    y_batch = [[tag2idx[t] for t in seq] for seq in labels[start_idx:end_idx]]
    
    # Pad sequences
    X_batch = pad_sequences(X_batch, maxlen=MAX_LEN, padding="post")
    y_batch = pad_sequences(y_batch, maxlen=MAX_LEN, padding="post")
    
    # Convert to categorical
    y_batch = [to_categorical(i, num_classes=len(tag2idx)) for i in y_batch]
    
    # Add to main arrays
    X.extend(X_batch)
    y.extend(y_batch)
    
    print(f"Processed batch {batch_idx+1}/{num_batches} ({end_idx}/{len(sentences)} sequences)")

# Convert to numpy arrays for efficiency
X = np.array(X)
y = np.array(y)

# Calculate sequence length statistics
lengths = [len(s) for s in sentences]
print("Longest name:", max(lengths))
print("95th percentile length:", np.percentile(lengths, 95))
print(f"Encoding completed in {time.time() - encoding_start_time:.2f} seconds")

# --------------------------
# Train/Test Split
# --------------------------
print("Performing train-test split...")
split_start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
print(f"Train set: {X_train.shape[0]} sequences, Test set: {X_test.shape[0]} sequences")
print(f"Split completed in {time.time() - split_start_time:.2f} seconds")

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
# Setup callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Try to enable mixed precision for faster training on compatible GPUs
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled: using float16 for faster training")
except:
    print("Mixed precision not available, using default precision")

print("Starting training...")
training_start_time = time.time()
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr, tensorboard_callback],
    verbose=1
)
print(f"Training completed in {time.time() - training_start_time:.2f} seconds")

# --------------------------
# Evaluation and Save
# --------------------------
print("Evaluating model...")
eval_start_time = time.time()
eval_results = model.evaluate(X_test, y_test, verbose=1)
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
