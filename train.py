import tensorflow as tf
import pandas as pd
import numpy as np
import json
import datetime
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense

# --------------------------
# Load Data
# --------------------------
start_time = time.time()
print("Starting data loading and processing...")

# Read the CSV file directly as text to handle empty lines properly
# Use a more efficient approach with generators to reduce memory usage
sentences = []
labels = []

def process_csv():
    with open("names_dataset.csv", "r", encoding="utf-8") as f:
        # Skip header
        next(f)
        
        current_sentence = []
        current_labels = []
        
        for line in f:
            line = line.strip()
            
            if not line:
                if current_sentence and current_labels:
                    yield current_sentence.copy(), current_labels.copy()
                    current_sentence = []
                    current_labels = []
            else:
                parts = line.split(",")
                if len(parts) == 2:
                    token, label = parts[0].strip(), parts[1].strip()
                    if token and label:
                        current_sentence.append(token)
                        current_labels.append(label)
        
        # Don't forget the last sequence
        if current_sentence and current_labels:
            yield current_sentence, current_labels

# Process in batches to improve performance
max_sequences = 1000000  # Limit to prevent memory issues
sequence_count = 0

for sent, label in process_csv():
    sentences.append(sent)
    labels.append(label)
    sequence_count += 1
    
    if sequence_count >= max_sequences:
        print(f"Reached maximum sequence limit ({max_sequences}). Truncating dataset.")
        break

print(f"Data loading completed in {time.time() - start_time:.2f} seconds")

print(f"Created {len(sentences)} sequences from dataset")

# --------------------------
# Encode
# --------------------------
print("Starting encoding process...")
encoding_start_time = time.time()

# More efficient vocabulary building
from collections import Counter

# Count word and tag frequencies
word_counter = Counter(token for sent in sentences for token in sent)
tag_counter = Counter(tag for seq in labels for tag in seq)

# Only keep words that appear at least twice to reduce vocabulary size
min_word_freq = 2
words = [word for word, count in word_counter.items() if count >= min_word_freq]
tags = list(tag_counter.keys())

print(f"Vocabulary size: {len(words)} words, {len(tags)} tags")

# Create mappings
word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["PAD"] = 0
word2idx["UNK"] = 1

tag2idx = {t: i for i, t in enumerate(tags)}

# Save mappings
json.dump(word2idx, open("word2idx.json", "w"))
json.dump(tag2idx, open("tag2idx.json", "w"))

# Process in batches to reduce memory pressure
print("Converting sequences to numeric format...")
batch_size = 250000
X = []
y = []

for i in range(0, len(sentences), batch_size):
    batch_sentences = sentences[i:i+batch_size]
    batch_labels = labels[i:i+batch_size]
    
    # Convert words to indices
    X.extend([[word2idx.get(w, 1) for w in s] for s in batch_sentences])
    y.extend([[tag2idx[t] for t in seq] for seq in batch_labels])

# Find maximum sequence length
MAX_LEN = min(max(len(s) for s in X), 50)  # Cap at 50 to prevent excessive padding
print(f"Maximum sequence length: {MAX_LEN}")

# Pad sequences
print("Padding sequences...")
X = pad_sequences(X, maxlen=MAX_LEN, padding="post")
y = pad_sequences(y, maxlen=MAX_LEN, padding="post")

# Convert to one-hot encoding
print("Converting to one-hot encoding...")
y = [to_categorical(i, num_classes=len(tag2idx)) for i in y]

print(f"Encoding completed in {time.time() - encoding_start_time:.2f} seconds")

print("🔢 Total sequences:", len(sentences))


# --------------------------
# Train/Test Split
# --------------------------
print("Performing train-test split...")
split_start_time = time.time()

# Convert y to numpy array for more efficient processing
y_array = np.array(y)

# Use a smaller test size to reduce training time
X_train, X_test, y_train, y_test = train_test_split(X, y_array, test_size=0.05, random_state=42)

print(f"Train set size: {len(X_train)} sequences")
print(f"Test set size: {len(X_test)} sequences")
print(f"Split completed in {time.time() - split_start_time:.2f} seconds")

# --------------------------
# Build Model
# --------------------------
# Check for GPU availability and configure memory growth to prevent OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU is available. Using GPU for training.")
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except:
            print("Memory growth setting failed.")
else:
    print("No GPU found. Using CPU for training.")

# Simplified model with fewer parameters
model = Sequential([
    Embedding(input_dim=len(word2idx), output_dim=32, input_length=MAX_LEN),  # Reduced embedding dimension
    Bidirectional(LSTM(32, return_sequences=True)),  # Reduced LSTM units
    TimeDistributed(Dense(len(tag2idx), activation="softmax"))
])

# Use a more efficient optimizer with a higher learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# --------------------------
# Train
# --------------------------
print("Starting model training...")
train_start_time = time.time()

# Add callbacks for early stopping and learning rate reduction
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=2, restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001
)

# Add TensorBoard callback for monitoring (optional)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, profile_batch=0
)

# Use mixed precision for faster training on compatible GPUs
try:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print('Mixed precision enabled')
except:
    print('Mixed precision not available')

# Increase batch size for faster training
history = model.fit(
    X_train, y_train, 
    batch_size=256,  # Further increased batch size
    epochs=5,  # Reduced epochs for faster training
    validation_data=(X_test, y_test),  # Use test data directly instead of validation_split
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print(f"Training completed in {time.time() - train_start_time:.2f} seconds")

# --------------------------
# Evaluate Model
# --------------------------
print("\nEvaluating model...")
eval_start_time = time.time()
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Evaluation completed in {time.time() - eval_start_time:.2f} seconds")

# --------------------------
# Save Model
# --------------------------
print("\nSaving model...")
save_start_time = time.time()

# Save model in Keras format
model.save("name_entity_model.keras")

print(f"Model saved in {time.time() - save_start_time:.2f} seconds")
print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")
