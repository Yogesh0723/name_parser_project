import tensorflow as tf
import pandas as pd
import numpy as np
import json
import datetime
import time
import unicodedata
import os
import glob
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
EMBED_DIM = 64  # Increased for better representation
LSTM_UNITS = 64  # Increased for better learning
BATCH_SIZE = 512  # Increased for better GPU utilization
EPOCHS = 10
MIN_WORD_FREQ = 2
DATASETS_DIR = "datasets"  # Directory containing regional datasets
MAX_SEQUENCES_PER_REGION = 100000  # Limit per region to balance dataset

# --------------------------
# Helpers
# --------------------------
def normalize_token(token):
    return unicodedata.normalize("NFKC", token).strip()

def find_dataset_files():
    """Find all regional dataset CSV files"""
    pattern = os.path.join(DATASETS_DIR, "names_dataset_*.csv")
    files = glob.glob(pattern)
    print(f"Found {len(files)} regional dataset files:")
    for file in files:
        file_size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
        print(f"  📁 {os.path.basename(file)} ({file_size:.1f} MB)")
    return files

def process_single_csv(file_path, max_sequences=None):
    """Process a single CSV file and return sentences and labels"""
    sentences, labels = [], []
    region_name = os.path.basename(file_path).replace("names_dataset_", "").replace(".csv", "")
    
    print(f"  📖 Processing {region_name} dataset...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        current_sentence, current_labels = [], []
        sequence_count = 0
        
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence.copy())
                    labels.append(current_labels.copy())
                    sequence_count += 1
                    current_sentence.clear()
                    current_labels.clear()
                    
                    if max_sequences and sequence_count >= max_sequences:
                        break
            else:
                parts = line.split(",")
                if len(parts) == 2:
                    token, label = normalize_token(parts[0]), parts[1].strip()
                    if token and label:
                        current_sentence.append(token)
                        current_labels.append(label)
        
        # Don't forget the last sequence
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)
            sequence_count += 1
    
    print(f"    ✅ Loaded {len(sentences)} sequences from {region_name}")
    return sentences, labels

def load_all_regional_datasets():
    """Load and combine all regional datasets"""
    dataset_files = find_dataset_files()
    
    if not dataset_files:
        print("❌ No dataset files found! Please run the dataset generator first.")
        print(f"   Looking for files in: {os.path.abspath(DATASETS_DIR)}")
        return [], []
    
    all_sentences, all_labels = [], []
    total_sequences = 0
    
    print(f"\n🚀 Loading datasets from {len(dataset_files)} regions...")
    print("=" * 60)
    
    for file_path in dataset_files:
        try:
            sentences, labels = process_single_csv(file_path, MAX_SEQUENCES_PER_REGION)
            all_sentences.extend(sentences)
            all_labels.extend(labels)
            total_sequences += len(sentences)
        except Exception as e:
            print(f"  ❌ Error processing {file_path}: {e}")
    
    print("=" * 60)
    print(f"🎉 Total loaded: {total_sequences} sequences from all regions")
    
    # Shuffle the combined dataset to mix regions
    combined = list(zip(all_sentences, all_labels))
    np.random.shuffle(combined)
    all_sentences, all_labels = zip(*combined)
    
    return list(all_sentences), list(all_labels)

# --------------------------
# Load Data
# --------------------------
print("🌍 Multi-Regional Name Entity Recognition Training")
print("=" * 60)
print("Starting data loading and processing...")
start_time = time.time()

# Load all regional datasets
sentences, labels = load_all_regional_datasets()

if not sentences:
    print("❌ No data loaded. Exiting...")
    exit(1)

print(f"\n📊 Dataset Statistics:")
print(f"   Total sequences: {len(sentences):,}")
print(f"   Average sequence length: {np.mean([len(s) for s in sentences]):.1f}")

# --------------------------
# Encoding
# --------------------------
print("\n🔧 Building vocabulary...")
encoding_start_time = time.time()

# Build vocabulary from all data
word_counter = Counter(token for sent in sentences for token in sent)
tag_counter = Counter(tag for seq in labels for tag in seq)

# Show tag distribution
print(f"\n📋 Tag distribution:")
for tag, count in tag_counter.most_common():
    print(f"   {tag}: {count:,}")

words = [w for w, c in word_counter.items() if c >= MIN_WORD_FREQ]
tags = list(tag_counter.keys())

word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["PAD"] = 0
word2idx["UNK"] = 1
tag2idx = {t: i for i, t in enumerate(tags)}

# Save mappings
os.makedirs("models", exist_ok=True)
json.dump(word2idx, open("models/word2idx.json", "w"))
json.dump(tag2idx, open("models/tag2idx.json", "w"))

print(f"   Vocabulary size: {len(word2idx):,}")
print(f"   Tag classes: {len(tag2idx)}")

# Process data in batches to reduce memory usage
print("\n🔄 Encoding sequences in batches...")
BATCH_SIZE_ENCODING = 50000
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
    
    progress = (batch_idx + 1) / num_batches * 100
    print(f"   Batch {batch_idx+1}/{num_batches} ({progress:.1f}%) - {end_idx:,}/{len(sentences):,} sequences")

# Convert to numpy arrays for efficiency
X = np.array(X)
y = np.array(y)

# Calculate sequence length statistics
lengths = [len(s) for s in sentences]
print(f"\n📏 Sequence length statistics:")
print(f"   Shortest: {min(lengths)}")
print(f"   Longest: {max(lengths)}")
print(f"   Average: {np.mean(lengths):.1f}")
print(f"   95th percentile: {np.percentile(lengths, 95):.1f}")
print(f"   Encoding completed in {time.time() - encoding_start_time:.2f} seconds")

# --------------------------
# Train/Test Split
# --------------------------
print("\n✂️ Performing train-test split...")
split_start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=None)
print(f"   Train set: {X_train.shape[0]:,} sequences")
print(f"   Test set: {X_test.shape[0]:,} sequences")
print(f"   Split completed in {time.time() - split_start_time:.2f} seconds")

# --------------------------
# Model - Enhanced Architecture
# --------------------------
print("\n🏗️ Building enhanced Bi-LSTM model...")

input_layer = Input(shape=(MAX_LEN,), name="Input")
mask = Masking(mask_value=0)(input_layer)
embed = Embedding(input_dim=len(word2idx), output_dim=EMBED_DIM, mask_zero=True)(mask)

# Enhanced architecture with dropout
bilstm1 = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(embed)
bilstm2 = Bidirectional(LSTM(LSTM_UNITS//2, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(bilstm1)

output = TimeDistributed(Dense(len(tag2idx), activation="softmax"))(bilstm2)
model = Model(inputs=input_layer, outputs=output, name="multi_regional_ner")

# Enhanced optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

print(f"   Model parameters: {model.count_params():,}")
model.summary()

# --------------------------
# Training with Enhanced Callbacks
# --------------------------
print("\n🎯 Setting up training...")

# Setup enhanced callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=2, 
        min_lr=0.00001,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='models/best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Setup TensorBoard
log_dir = f"logs/multi_regional_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
callbacks.append(tensorboard_callback)

# Try to enable mixed precision for faster training
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("   ⚡ Mixed precision enabled for faster training")
except:
    print("   ⚠️ Mixed precision not available, using default precision")

print(f"\n🚀 Starting training for {EPOCHS} epochs...")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Training samples: {len(X_train):,}")
print(f"   Validation samples: {len(X_test):,}")
print("=" * 60)

training_start_time = time.time()
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)
training_time = time.time() - training_start_time
print(f"✅ Training completed in {training_time/60:.1f} minutes")

# --------------------------
# Evaluation and Save
# --------------------------
print("\n📊 Evaluating model...")
eval_start_time = time.time()
eval_results = model.evaluate(X_test, y_test, verbose=1)
print(f"   Evaluation completed in {time.time() - eval_start_time:.2f} seconds")

print("\n💾 Saving final model...")
save_start_time = time.time()
model.save("models/multi_regional_ner_model.keras")
print(f"   Model saved in {time.time() - save_start_time:.2f} seconds")

# Save training history
history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
json.dump(history_dict, open("models/training_history.json", "w"))

# --------------------------
# Final Summary
# --------------------------
total_time = time.time() - start_time
print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE!")
print("=" * 60)
print(f"📈 Final Results:")
print(f"   Test Loss: {eval_results[0]:.4f}")
print(f"   Test Accuracy: {eval_results[1]:.4f}")
print(f"\n⏱️ Time Summary:")
print(f"   Total time: {total_time/60:.1f} minutes")
print(f"   Data loading: {encoding_start_time - start_time:.1f}s")
print(f"   Encoding: {time.time() - encoding_start_time - training_time:.1f}s")
print(f"   Training: {training_time/60:.1f} minutes")
print(f"\n📊 Dataset Summary:")
print(f"   Total sequences: {len(sentences):,}")
print(f"   Vocabulary size: {len(word2idx):,}")
print(f"   Tag classes: {len(tag2idx)}")
print(f"   Regions trained: {len(find_dataset_files())}")
print(f"\n📁 Saved Files:")
print(f"   Model: models/multi_regional_ner_model.keras")
print(f"   Best model: models/best_model.keras")
print(f"   Vocabularies: models/word2idx.json, models/tag2idx.json")
print(f"   History: models/training_history.json")
print(f"   TensorBoard logs: {log_dir}")
print("\n🚀 Ready for inference!")