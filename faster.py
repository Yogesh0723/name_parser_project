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
import os
import multiprocessing as mp

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

def process_csv():
    with open("names_dataset.csv", "r", encoding="utf-8") as f:
        next(f)
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

def encode_batch(sentences, labels, word2idx, tag2idx):
    X_batch = [[word2idx.get(w, 1) for w in s] for s in sentences]
    y_batch = [[tag2idx[t] for t in seq] for seq in labels]

    X_batch = pad_sequences(X_batch, maxlen=MAX_LEN, padding="post")
    y_batch = pad_sequences(y_batch, maxlen=MAX_LEN, padding="post")
    y_batch = [to_categorical(i, num_classes=len(tag2idx)) for i in y_batch]

    return X_batch, y_batch

# --------------------------
# Load or Preprocess Data
# --------------------------
print("Starting data loading and processing...")
start_time = time.time()

X_FILE, Y_FILE = "X.npy", "y.npy"

if os.path.exists(X_FILE) and os.path.exists(Y_FILE):
    print("📦 Loading preprocessed arrays...")
    X = np.load(X_FILE)
    y = np.load(Y_FILE)
    with open("word2idx.json") as f:
        word2idx = json.load(f)
    with open("tag2idx.json") as f:
        tag2idx = json.load(f)
else:
    print("🛠 Preprocessing from CSV...")
    raw_data = list(process_csv())
    sentences, labels = zip(*raw_data)

    print(f"Loaded {len(sentences)} sequences")

    print("Building vocabulary...")
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

    print("Encoding sequences in parallel batches...")
    BATCH_SIZE_ENCODING = 250000
    num_batches = (len(sentences) + BATCH_SIZE_ENCODING - 1) // BATCH_SIZE_ENCODING

    def process_batch(idx):
        start = idx * BATCH_SIZE_ENCODING
        end = min((idx + 1) * BATCH_SIZE_ENCODING, len(sentences))
        return encode_batch(sentences[start:end], labels[start:end], word2idx, tag2idx)

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_batch, range(num_batches))

    X, y = zip(*results)
    X = np.vstack(X)
    y = np.vstack(y)

    np.save(X_FILE, X)
    np.save(Y_FILE, y)

print("Train/Test Split...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

print("Building model...")
input_layer = Input(shape=(MAX_LEN,), name="Input")
mask = Masking(mask_value=0)(input_layer)
embed = Embedding(input_dim=len(word2idx), output_dim=EMBED_DIM)(mask)
bilstm = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(embed)
output = TimeDistributed(Dense(len(tag2idx), activation="softmax"))(bilstm)
model = Model(inputs=input_layer, outputs=output, name="functional")

optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")
except:
    print("Default precision active")

print("Training...")
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr, tensorboard_callback],
    verbose=1
)

print("Evaluating...")
eval_results = model.evaluate(X_test, y_test, verbose=1)
model.save("name_entity_functional_model.keras")

print("Final Accuracy:", eval_results[1])
