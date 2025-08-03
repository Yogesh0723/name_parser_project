import tensorflow as tf
import pandas as pd
import numpy as np
import json
import datetime
import time
import unicodedata
import random
import os
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, GRU, TimeDistributed, 
    Dense, Masking, Dropout, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Concatenate, Conv1D, GlobalMaxPooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow Addons for CRF (optional)
try:
    import tensorflow_addons as tfa
    CRF_AVAILABLE = True
    print("✅ TensorFlow Addons available - CRF layer enabled")
except ImportError:
    CRF_AVAILABLE = False
    print("⚠️  TensorFlow Addons not found - install with: pip install tensorflow-addons")

# --------------------------
# Enhanced Configuration
# --------------------------
class Config:
    # Data parameters
    MAX_LEN = 30
    CHAR_MAX_LEN = 15
    MIN_WORD_FREQ = 2
    MAX_SEQUENCES = 100000
    TEST_SIZE = 0.1
    VALIDATION_SIZE = 0.1
    
    # Model architecture options
    MODEL_TYPE = "transformer"  # Options: "bilstm", "cnn_bilstm", "transformer", "ensemble"
    USE_CHAR_FEATURES = True
    USE_CRF = CRF_AVAILABLE
    
    # Architecture parameters
    EMBED_DIM = 128
    CHAR_EMBED_DIM = 32
    LSTM_UNITS = 128
    NUM_HEADS = 8
    FF_DIM = 256
    DROPOUT_RATE = 0.3
    L2_REG = 1e-4
    
    # Training parameters
    BATCH_SIZE = 128
    EPOCHS = 20
    LEARNING_RATE = 0.001
    WARMUP_STEPS = 1000
    PATIENCE = 5
    
    # Data augmentation
    AUGMENT_DATA = True
    AUGMENTATION_RATIO = 0.2
    
    # Performance optimizations
    USE_MIXED_PRECISION = True
    USE_CLASS_WEIGHTS = True

config = Config()

# --------------------------
# Enhanced Data Processing
# --------------------------
def normalize_token(token):
    """Enhanced token normalization"""
    token = unicodedata.normalize("NFKC", token).strip()
    return token.lower() if token.isalpha() else token

def create_char_vocab(sentences):
    """Create character-level vocabulary"""
    char_counter = Counter()
    for sent in sentences:
        for word in sent:
            char_counter.update(word.lower())
    
    chars = [c for c, count in char_counter.most_common() if count >= 2]
    char2idx = {c: i + 2 for i, c in enumerate(chars)}
    char2idx["PAD"] = 0
    char2idx["UNK"] = 1
    return char2idx

def advanced_augmentation(sentences, labels, ratio=0.2):
    """Advanced data augmentation techniques"""
    augmented_sentences = []
    augmented_labels = []
    
    # Title synonyms for replacement
    title_synonyms = {
        'Dr.': ['Dr', 'Doctor', 'DR.', 'DR'],
        'Mr.': ['Mr', 'MR.', 'MR', 'Mister'],
        'Mrs.': ['Mrs', 'MRS.', 'MRS'],
        'Ms.': ['Ms', 'MS.', 'MS'],
        'Prof.': ['Prof', 'Professor', 'PROF.']
    }
    
    for sent, lbls in zip(sentences, labels):
        # Original data
        augmented_sentences.append(sent)
        augmented_labels.append(lbls)
        
        if random.random() < ratio:
            # 1. Case variation augmentation
            if random.random() < 0.4:
                aug_sent = []
                for token in sent:
                    if random.random() < 0.3:
                        aug_sent.append(token.upper())
                    elif random.random() < 0.3:
                        aug_sent.append(token.lower())
                    else:
                        aug_sent.append(token)
                augmented_sentences.append(aug_sent)
                augmented_labels.append(lbls)
            
            # 2. Synonym replacement
            if random.random() < 0.3:
                aug_sent = []
                for token in sent:
                    if token in title_synonyms and random.random() < 0.5:
                        aug_sent.append(random.choice(title_synonyms[token]))
                    else:
                        aug_sent.append(token)
                if aug_sent != sent:
                    augmented_sentences.append(aug_sent)
                    augmented_labels.append(lbls)
            
            # 3. Minor typo simulation (only for non-entity tokens)
            if random.random() < 0.2:
                aug_sent = []
                for i, token in enumerate(sent):
                    if lbls[i] == 'O' and len(token) > 3 and random.random() < 0.3:
                        # Introduce minor typos
                        chars = list(token)
                        if len(chars) > 1 and random.random() < 0.5:
                            # Swap adjacent characters
                            pos = random.randint(0, len(chars)-2)
                            chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
                        aug_sent.append(''.join(chars))
                    else:
                        aug_sent.append(token)
                augmented_sentences.append(aug_sent)
                augmented_labels.append(lbls)
    
    print(f"🔄 Data augmentation: {len(sentences)} → {len(augmented_sentences)} sequences")
    return augmented_sentences, augmented_labels

def create_char_features(sentences, char2idx, max_char_len=15):
    """Create character-level features"""
    char_features = []
    for sent in sentences:
        sent_chars = []
        for word in sent:
            word_chars = [char2idx.get(c, 1) for c in word.lower()[:max_char_len]]
            word_chars = word_chars + [0] * (max_char_len - len(word_chars))
            sent_chars.append(word_chars)
        char_features.append(sent_chars)
    return char_features

def compute_class_weights(labels, tag2idx):
    """Compute class weights for imbalanced data"""
    if not config.USE_CLASS_WEIGHTS:
        return None
    
    # Flatten all labels
    all_labels = [label for seq in labels for label in seq]
    unique_labels = list(set(all_labels))
    
    # Compute weights
    weights = compute_class_weight(
        'balanced',
        classes=unique_labels,
        y=all_labels
    )
    
    # Convert to dictionary with tag indices
    class_weights = {}
    for label, weight in zip(unique_labels, weights):
        if label in tag2idx:
            class_weights[tag2idx[label]] = weight
    
    print(f"📊 Computed class weights for {len(class_weights)} classes")
    return class_weights

# --------------------------
# Advanced Model Architectures
# --------------------------
class TransformerBlock(tf.keras.layers.Layer):
    """Enhanced Transformer encoder block"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dropout(rate),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
        })
        return config

def build_enhanced_bilstm_model(vocab_size, char_vocab_size, tag_size, config):
    """Enhanced BiLSTM model with character features and attention"""
    # Word input
    word_input = Input(shape=(config.MAX_LEN,), name="word_input")
    word_mask = Masking(mask_value=0)(word_input)
    
    # Word embeddings
    word_embed = Embedding(
        input_dim=vocab_size, 
        output_dim=config.EMBED_DIM,
        embeddings_regularizer=l2(config.L2_REG),
        mask_zero=True
    )(word_mask)
    
    # Character-level features (if enabled)
    if config.USE_CHAR_FEATURES and char_vocab_size:
        char_input = Input(shape=(config.MAX_LEN, config.CHAR_MAX_LEN), name="char_input")
        char_embed = TimeDistributed(
            Embedding(char_vocab_size, config.CHAR_EMBED_DIM, mask_zero=True)
        )(char_input)
        char_conv = TimeDistributed(Conv1D(64, 3, activation='relu', padding='same'))(char_embed)
        char_pool = TimeDistributed(GlobalMaxPooling1D())(char_conv)
        
        # Combine word and character features
        combined_embed = Concatenate()([word_embed, char_pool])
        inputs = [word_input, char_input]
    else:
        combined_embed = word_embed
        inputs = word_input
    
    combined_embed = Dropout(config.DROPOUT_RATE)(combined_embed)
    
    # Multiple BiLSTM layers with residual connections
    lstm1 = Bidirectional(LSTM(
        config.LSTM_UNITS, 
        return_sequences=True, 
        dropout=0.2, 
        recurrent_dropout=0.2
    ))(combined_embed)
    lstm1 = LayerNormalization()(lstm1)
    
    lstm2 = Bidirectional(LSTM(
        config.LSTM_UNITS//2, 
        return_sequences=True, 
        dropout=0.2, 
        recurrent_dropout=0.2
    ))(lstm1)
    lstm2 = LayerNormalization()(lstm2)
    
    # Self-attention mechanism
    attention = MultiHeadAttention(
        num_heads=4, 
        key_dim=config.LSTM_UNITS
    )(lstm2, lstm2)
    attention = Dropout(config.DROPOUT_RATE)(attention)
    
    # Combine LSTM and attention
    combined = Concatenate()([lstm2, attention])
    combined = Dense(
        config.LSTM_UNITS, 
        activation='relu', 
        kernel_regularizer=l2(config.L2_REG)
    )(combined)
    combined = Dropout(config.DROPOUT_RATE)(combined)
    
    # Output layer
    if config.USE_CRF and CRF_AVAILABLE:
        # CRF layer for better sequence labeling
        dense_output = TimeDistributed(Dense(tag_size))(combined)
        crf = tfa.layers.CRF(tag_size, sparse_target=False)
        output = crf(dense_output)
        
        model = Model(inputs=inputs, outputs=output, name="enhanced_bilstm_crf")
        
        # Special compilation for CRF
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss=crf.loss, metrics=[crf.accuracy])
    else:
        # Standard softmax output
        output = TimeDistributed(Dense(tag_size, activation="softmax"))(combined)
        model = Model(inputs=inputs, outputs=output, name="enhanced_bilstm")
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model

def build_transformer_model(vocab_size, char_vocab_size, tag_size, config):
    """Transformer-based NER model"""
    # Word input
    word_input = Input(shape=(config.MAX_LEN,), name="word_input")
    
    # Word embeddings with positional encoding
    word_embed = Embedding(
        input_dim=vocab_size, 
        output_dim=config.EMBED_DIM,
        mask_zero=True
    )(word_input)
    
    # Character features (if enabled)
    if config.USE_CHAR_FEATURES and char_vocab_size:
        char_input = Input(shape=(config.MAX_LEN, config.CHAR_MAX_LEN), name="char_input")
        char_embed = TimeDistributed(
            Embedding(char_vocab_size, config.CHAR_EMBED_DIM, mask_zero=True)
        )(char_input)
        char_conv = TimeDistributed(Conv1D(64, 3, activation='relu', padding='same'))(char_embed)
        char_pool = TimeDistributed(GlobalMaxPooling1D())(char_conv)
        
        combined_embed = Concatenate()([word_embed, char_pool])
        inputs = [word_input, char_input]
    else:
        combined_embed = word_embed
        inputs = word_input
    
    # Positional encoding (simple version)
    seq_len = tf.shape(combined_embed)[1]
    positions = tf.range(start=0, limit=seq_len, dtype=tf.float32)
    pos_embed = Embedding(config.MAX_LEN, config.EMBED_DIM)(positions)
    combined_embed = combined_embed + pos_embed
    
    combined_embed = Dropout(config.DROPOUT_RATE)(combined_embed)
    
    # Multiple Transformer blocks
    transformer_out = combined_embed
    for i in range(3):  # 3 transformer layers
        transformer_out = TransformerBlock(
            embed_dim=config.EMBED_DIM,
            num_heads=config.NUM_HEADS,
            ff_dim=config.FF_DIM,
            rate=config.DROPOUT_RATE
        )(transformer_out)
    
    # Output layer
    if config.USE_CRF and CRF_AVAILABLE:
        dense_output = TimeDistributed(Dense(tag_size))(transformer_out)
        crf = tfa.layers.CRF(tag_size, sparse_target=False)
        output = crf(dense_output)
        
        model = Model(inputs=inputs, outputs=output, name="transformer_crf")
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss=crf.loss, metrics=[crf.accuracy])
    else:
        output = TimeDistributed(Dense(tag_size, activation="softmax"))(transformer_out)
        model = Model(inputs=inputs, outputs=output, name="transformer")
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model

def build_cnn_bilstm_model(vocab_size, char_vocab_size, tag_size, config):
    """CNN + BiLSTM hybrid model"""
    word_input = Input(shape=(config.MAX_LEN,), name="word_input")
    word_embed = Embedding(
        input_dim=vocab_size, 
        output_dim=config.EMBED_DIM,
        mask_zero=True
    )(word_input)
    
    # Character features
    if config.USE_CHAR_FEATURES and char_vocab_size:
        char_input = Input(shape=(config.MAX_LEN, config.CHAR_MAX_LEN), name="char_input")
        char_embed = TimeDistributed(
            Embedding(char_vocab_size, config.CHAR_EMBED_DIM, mask_zero=True)
        )(char_input)
        char_conv = TimeDistributed(Conv1D(64, 3, activation='relu', padding='same'))(char_embed)
        char_pool = TimeDistributed(GlobalMaxPooling1D())(char_conv)
        
        combined_embed = Concatenate()([word_embed, char_pool])
        inputs = [word_input, char_input]
    else:
        combined_embed = word_embed
        inputs = word_input
    
    # CNN features with multiple filter sizes
    conv_outputs = []
    for filter_size in [2, 3, 4, 5]:
        conv = Conv1D(
            filters=64, 
            kernel_size=filter_size, 
            activation='relu', 
            padding='same'
        )(combined_embed)
        conv_outputs.append(conv)
    
    # Combine CNN features
    cnn_combined = Concatenate()(conv_outputs)
    cnn_combined = Dropout(config.DROPOUT_RATE)(cnn_combined)
    
    # BiLSTM layers
    bilstm = Bidirectional(LSTM(
        config.LSTM_UNITS, 
        return_sequences=True, 
        dropout=0.2
    ))(cnn_combined)
    bilstm = LayerNormalization()(bilstm)
    
    # Final processing
    final_layer = Dense(
        config.LSTM_UNITS, 
        activation='relu',
        kernel_regularizer=l2(config.L2_REG)
    )(bilstm)
    final_layer = Dropout(config.DROPOUT_RATE)(final_layer)
    
    # Output
    if config.USE_CRF and CRF_AVAILABLE:
        dense_output = TimeDistributed(Dense(tag_size))(final_layer)
        crf = tfa.layers.CRF(tag_size, sparse_target=False)
        output = crf(dense_output)
        
        model = Model(inputs=inputs, outputs=output, name="cnn_bilstm_crf")
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss=crf.loss, metrics=[crf.accuracy])
    else:
        output = TimeDistributed(Dense(tag_size, activation="softmax"))(final_layer)
        model = Model(inputs=inputs, outputs=output, name="cnn_bilstm")
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model

# --------------------------
# Advanced Learning Rate Scheduling
# --------------------------
def create_lr_schedule(total_steps, warmup_steps=1000, initial_lr=0.001):
    """Warmup + Cosine decay learning rate schedule"""
    def lr_schedule(step):
        step = tf.cast(step, tf.float32)
        warmup_steps_f = tf.cast(warmup_steps, tf.float32)
        total_steps_f = tf.cast(total_steps, tf.float32)
        
        if step < warmup_steps_f:
            return (step / warmup_steps_f) * initial_lr
        else:
            progress = (step - warmup_steps_f) / (total_steps_f - warmup_steps_f)
            return 0.5 * initial_lr * (1 + tf.cos(np.pi * progress))
    
    return lr_schedule

# --------------------------
# Advanced Monitoring Callback
# --------------------------
class AdvancedMonitoring(tf.keras.callbacks.Callback):
    """Advanced monitoring with detailed metrics"""
    def __init__(self, validation_data, tag2idx, patience=5):
        super().__init__()
        self.validation_data = validation_data
        self.tag2idx = tag2idx
        self.idx2tag = {v: k for k, v in tag2idx.items()}
        self.best_f1 = 0
        self.patience = patience
        self.wait = 0
        
    def on_epoch_end(self, epoch, logs=None):
        if config.USE_CRF:
            # For CRF models, use the built-in metrics
            val_acc = logs.get('val_crf_accuracy', 0)
            print(f"\n🎯 Epoch {epoch+1} - Validation CRF Accuracy: {val_acc:.4f}")
            
            if val_acc > self.best_f1:
                self.best_f1 = val_acc
                self.model.save('best_model.keras')
                print(f"💾 New best model saved: {val_acc:.4f}")
                self.wait = 0
            else:
                self.wait += 1
                
        else:
            # For non-CRF models, calculate custom metrics
            X_val, y_val = self.validation_data
            predictions = self.model.predict(X_val, verbose=0)
            
            # Convert predictions to labels
            y_pred = np.argmax(predictions, axis=-1)
            y_true = np.argmax(y_val, axis=-1)
            
            # Calculate accuracy ignoring padding
            mask = y_true != 0  # Ignore padding tokens
            accuracy = np.mean(y_pred[mask] == y_true[mask])
            
            print(f"\n🎯 Epoch {epoch+1} - Validation Accuracy: {accuracy:.4f}")
            
            if accuracy > self.best_f1:
                self.best_f1 = accuracy
                self.model.save('best_model.keras')
                print(f"💾 New best model saved: {accuracy:.4f}")
                self.wait = 0
            else:
                self.wait += 1
        
        # Early stopping
        if self.wait >= self.patience:
            print(f"\n⏹️  Early stopping after {epoch+1} epochs")
            self.model.stop_training = True

# --------------------------
# Main Training Pipeline
# --------------------------
def main():
    print("🚀 Starting Enhanced NER Model Training...")
    print(f"📋 Configuration: {config.MODEL_TYPE} model")
    print(f"   - CRF: {'✅' if config.USE_CRF else '❌'}")
    print(f"   - Char features: {'✅' if config.USE_CHAR_FEATURES else '❌'}")
    print(f"   - Data augmentation: {'✅' if config.AUGMENT_DATA else '❌'}")
    
    start_time = time.time()
    
    # Enable optimizations
    if config.USE_MIXED_PRECISION:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision enabled")
        except:
            print("⚠️  Mixed precision not available")
    
    # Load and process data
    print("\n📊 Loading and processing data...")
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
    
    for i, (sent, lbl) in enumerate(process_csv()):
        sentences.append(sent)
        labels.append(lbl)
        if i + 1 >= config.MAX_SEQUENCES:
            break
    
    print(f"📈 Loaded {len(sentences)} sequences")
    
    # Data augmentation
    if config.AUGMENT_DATA:
        sentences, labels = advanced_augmentation(sentences, labels, config.AUGMENTATION_RATIO)
    
    # Build vocabularies
    print("🔤 Building vocabularies...")
    word_counter = Counter(token for sent in sentences for token in sent)
    tag_counter = Counter(tag for seq in labels for tag in seq)
    
    words = [w for w, c in word_counter.items() if c >= config.MIN_WORD_FREQ]
    tags = list(tag_counter.keys())
    
    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx["PAD"] = 0
    word2idx["UNK"] = 1
    tag2idx = {t: i for i, t in enumerate(tags)}
    
    # Character vocabulary (if needed)
    char2idx = None
    char_vocab_size = 0
    if config.USE_CHAR_FEATURES:
        char2idx = create_char_vocab(sentences)
        char_vocab_size = len(char2idx)
        print(f"🔤 Character vocabulary size: {char_vocab_size}")
    
    # Save vocabularies
    json.dump(word2idx, open("word2idx.json", "w"))
    json.dump(tag2idx, open("tag2idx.json", "w"))
    if char2idx:
        json.dump(char2idx, open("char2idx.json", "w"))
    
    print(f"📚 Vocabularies - Words: {len(word2idx)}, Tags: {len(tag2idx)}")
    
    # Encode sequences
    print("🔢 Encoding sequences...")
    X_words = [[word2idx.get(w, 1) for w in s] for s in sentences]
    y = [[tag2idx[t] for t in seq] for seq in labels]
    
    # Pad sequences
    X_words = pad_sequences(X_words, maxlen=config.MAX_LEN, padding="post")
    y = pad_sequences(y, maxlen=config.MAX_LEN, padding="post")
    
    # Character features
    X_chars = None
    if config.USE_CHAR_FEATURES:
        char_features = create_char_features(sentences, char2idx, config.CHAR_MAX_LEN)
        X_chars = pad_sequences([pad_sequences(seq, maxlen=config.CHAR_MAX_LEN, padding="post") 
                                for seq in char_features], maxlen=config.MAX_LEN, padding="post")
    
    # Convert labels (handle CRF vs standard)
    if not config.USE_CRF:
        y = np.array([to_categorical(seq, num_classes=len(tag2idx)) for seq in y])
    else:
        y = np.array(y)
    
    # Train/validation/test split
    print("✂️  Splitting data...")
    if config.USE_CHAR_FEATURES:
        X_train_w, X_temp_w, X_train_c, X_temp_c, y_train, y_temp = train_test_split(
            X_words, X_chars, y, test_size=config.TEST_SIZE + config.VALIDATION_SIZE, random_state=42
        )
        X_val_w, X_test_w, X_val_c, X_test_c, y_val, y_test = train_test_split(
            X_temp_w, X_temp_c, y_temp, test_size=config.TEST_SIZE/(config.TEST_SIZE + config.VALIDATION_SIZE), random_state=42
        )
        X_train = [X_train_w, X_train_c]
        X_val = [X_val_w, X_val_c]
        X_test = [X_test_w, X_test_c]
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_words, y, test_size=config.TEST_SIZE + config.VALIDATION_SIZE, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=config.TEST_SIZE/(config.TEST_SIZE + config.VALIDATION_SIZE), random_state=42
        )
    
    print(f"📊 Data split - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    # Compute class weights
    if config.USE_CLASS_WEIGHTS and not config.USE_CRF:
        class_weights = compute_class_weights([labels[i] for i in range(len(y_train))], tag2idx)
    else:
        class_weights = None
    
    # Build model
    print(f"🏗️  Building {config.MODEL_TYPE} model...")
    if config.MODEL_TYPE == "bilstm":
        model = build_enhanced_bilstm_model(len(word2idx), char_vocab_size, len(tag2idx), config)
    elif config.MODEL_TYPE == "transformer":
        model = build_transformer_model(len(word2idx), char_vocab_size, len(tag2idx), config)
    elif config.MODEL_TYPE == "cnn_bilstm":
        model = build_cnn_bilstm_model(len(word2idx), char_vocab_size, len(tag2idx), config)
    else:
        raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")
    
    model.summary()
    
    # Setup callbacks
    print("⚙️  Setting up training callbacks...")
    callbacks = []
    
    # Advanced monitoring
    monitor_callback = AdvancedMonitoring(
        validation_data=(X_val, y_val),
        tag2idx=tag2idx,
        patience=config.PATIENCE
    )
    callbacks.append(monitor_callback)
    
    # Learning rate scheduling
    total_steps = len(y_train) // config.BATCH_SIZE * config.EPOCHS
    lr_schedule = create_lr_schedule(total_steps, config.WARMUP_STEPS, config.LEARNING_RATE)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
    callbacks.append(lr_callback)
    
    # Reduce LR on plateau (backup)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )
    callbacks.append(reduce_lr)
    
    # TensorBoard logging
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks.append(tensorboard_callback)
    
    # Model checkpointing
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        'model_checkpoint.keras',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Training
    print("🚀 Starting training...")
    training_start_time = time.time()
    
    # Prepare training arguments
    fit_kwargs = {
        'x': X_train,
        'y': y_train,
        'batch_size': config.BATCH_SIZE,
        'epochs': config.EPOCHS,
        'validation_data': (X_val, y_val),
        'callbacks': callbacks,
        'verbose': 1
    }
    
    # Add class weights if applicable
    if class_weights and not config.USE_CRF:
        fit_kwargs['class_weight'] = class_weights
    
    history = model.fit(**fit_kwargs)
    
    print(f"✅ Training completed in {time.time() - training_start_time:.2f} seconds")
    
    # Load best model for evaluation
    try:
        model = tf.keras.models.load_model('best_model.keras', custom_objects={'TransformerBlock': TransformerBlock})
        print("📂 Loaded best model for evaluation")
    except:
        print("⚠️  Using last epoch model for evaluation")
    
    # Evaluation
    print("\n📊 Evaluating model...")
    eval_start_time = time.time()
    
    if config.USE_CRF:
        # CRF evaluation
        eval_results = model.evaluate(X_test, y_test, verbose=1)
        print(f"🎯 Test CRF Accuracy: {eval_results[1]:.4f}")
        final_accuracy = eval_results[1]
    else:
        # Standard evaluation with detailed metrics
        eval_results = model.evaluate(X_test, y_test, verbose=1)
        predictions = model.predict(X_test, verbose=0)
        
        # Convert predictions to labels
        y_pred = np.argmax(predictions, axis=-1)
        y_true = np.argmax(y_test, axis=-1)
        
        # Calculate metrics ignoring padding
        mask = y_true != 0
        accuracy = np.mean(y_pred[mask] == y_true[mask])
        final_accuracy = accuracy
        
        print(f"🎯 Test Accuracy: {accuracy:.4f}")
        
        # Detailed per-class metrics
        try:
            # Convert to label sequences for detailed analysis
            idx2tag = {v: k for k, v in tag2idx.items()}
            true_seqs = []
            pred_seqs = []
            
            for i in range(len(y_true)):
                true_seq = [idx2tag[idx] for idx in y_true[i] if idx != 0]
                pred_seq = [idx2tag[idx] for idx in y_pred[i] if idx != 0]
                # Ensure sequences have same length
                min_len = min(len(true_seq), len(pred_seq))
                if min_len > 0:
                    true_seqs.append(true_seq[:min_len])
                    pred_seqs.append(pred_seq[:min_len])
            
            # Per-tag analysis
            print("\n📈 Per-Tag Performance Analysis:")
            tag_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
            
            for true_seq, pred_seq in zip(true_seqs, pred_seqs):
                for true_tag, pred_tag in zip(true_seq, pred_seq):
                    if true_tag == pred_tag and true_tag != 'O':
                        tag_metrics[true_tag]['tp'] += 1
                    elif pred_tag != 'O':
                        tag_metrics[pred_tag]['fp'] += 1
                        if true_tag != 'O':
                            tag_metrics[true_tag]['fn'] += 1
                    elif true_tag != 'O':
                        tag_metrics[true_tag]['fn'] += 1
            
            print("   Tag        Precision  Recall     F1-Score   Support")
            print("   " + "-" * 55)
            overall_tp = overall_fp = overall_fn = 0
            
            for tag in sorted(tag_metrics.keys()):
                metrics = tag_metrics[tag]
                tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
                
                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / (tp + fn) if tp + fn > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                support = tp + fn
                
                print(f"   {tag:<10} {precision:.3f}      {recall:.3f}      {f1:.3f}      {support}")
                
                overall_tp += tp
                overall_fp += fp
                overall_fn += fn
            
            # Overall metrics
            overall_precision = overall_tp / (overall_tp + overall_fp) if overall_tp + overall_fp > 0 else 0
            overall_recall = overall_tp / (overall_tp + overall_fn) if overall_tp + overall_fn > 0 else 0
            overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if overall_precision + overall_recall > 0 else 0
            
            print("   " + "-" * 55)
            print(f"   Overall    {overall_precision:.3f}      {overall_recall:.3f}      {overall_f1:.3f}      {overall_tp + overall_fn}")
            
        except Exception as e:
            print(f"⚠️  Detailed metrics calculation failed: {e}")
    
    print(f"⏱️  Evaluation completed in {time.time() - eval_start_time:.2f} seconds")
    
    # Save final model and artifacts
    print("\n💾 Saving model and artifacts...")
    save_start_time = time.time()
    
    model_name = f"enhanced_ner_{config.MODEL_TYPE}_model.keras"
    model.save(model_name)
    
    # Save training history
    history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
    with open("training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)
    
    # Save configuration
    config_dict = {
        'model_type': config.MODEL_TYPE,
        'max_len': config.MAX_LEN,
        'embed_dim': config.EMBED_DIM,
        'lstm_units': config.LSTM_UNITS,
        'dropout_rate': config.DROPOUT_RATE,
        'learning_rate': config.LEARNING_RATE,
        'batch_size': config.BATCH_SIZE,
        'epochs': config.EPOCHS,
        'use_crf': config.USE_CRF,
        'use_char_features': config.USE_CHAR_FEATURES,
        'vocab_size': len(word2idx),
        'tag_size': len(tag2idx),
        'char_vocab_size': char_vocab_size,
        'final_accuracy': float(final_accuracy)
    }
    
    with open("model_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"💾 Model saved as: {model_name}")
    print(f"⏱️  Saving completed in {time.time() - save_start_time:.2f} seconds")
    
    # Summary
    print(f"\n🎉 Training Complete!")
    print(f"⏱️  Total time: {time.time() - start_time:.2f} seconds")
    print(f"📊 Dataset: {len(sentences)} sequences")
    print(f"🔤 Vocabulary: {len(word2idx)} words, {len(tag2idx)} tags")
    if config.USE_CHAR_FEATURES:
        print(f"🔤 Character vocab: {char_vocab_size} characters")
    print(f"🎯 Final accuracy: {final_accuracy:.4f}")
    print(f"💾 Model saved: {model_name}")
    
    # Training recommendations
    print(f"\n💡 Training Insights:")
    if final_accuracy < 0.85:
        print("   • Consider increasing model complexity or training time")
        print("   • Check data quality and label consistency")
        print("   • Try different model architecture (transformer vs bilstm)")
    elif final_accuracy > 0.95:
        print("   • Excellent performance! Consider model compression for production")
        print("   • Validate on additional test sets to ensure generalization")
    else:
        print("   • Good performance! Consider fine-tuning hyperparameters")
        print("   • Try ensemble methods for further improvement")
    
    if config.USE_CRF:
        print("   • CRF layer enabled for better sequence consistency")
    if config.USE_CHAR_FEATURES:
        print("   • Character features enabled for better morphological understanding")
    
    return model, word2idx, tag2idx, char2idx, history

# --------------------------
# Utility Functions for Production
# --------------------------
def load_trained_model(model_path, config_path=None):
    """Load a trained model with proper custom objects"""
    custom_objects = {'TransformerBlock': TransformerBlock}
    
    if CRF_AVAILABLE:
        custom_objects['CRF'] = tfa.layers.CRF
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    if config_path:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return model, config_dict
    
    return model

def predict_entities(model, sentence, word2idx, char2idx, tag2idx, config):
    """Predict entities for a single sentence"""
    # Tokenize and normalize
    tokens = sentence.split()  # Simple tokenization
    tokens = [normalize_token(token) for token in tokens]
    
    # Encode words
    word_ids = [word2idx.get(token, 1) for token in tokens]
    word_seq = pad_sequences([word_ids], maxlen=config.MAX_LEN, padding="post")
    
    # Encode characters (if needed)
    if config.USE_CHAR_FEATURES and char2idx:
        char_ids = []
        for token in tokens:
            token_chars = [char2idx.get(c, 1) for c in token.lower()[:config.CHAR_MAX_LEN]]
            token_chars = token_chars + [0] * (config.CHAR_MAX_LEN - len(token_chars))
            char_ids.append(token_chars)
        
        char_seq = pad_sequences([char_ids], maxlen=config.MAX_LEN, padding="post")
        X = [word_seq, char_seq]
    else:
        X = word_seq
    
    # Predict
    predictions = model.predict(X, verbose=0)
    
    if config.USE_CRF:
        # CRF output is already class indices
        pred_ids = predictions[0]
    else:
        # Softmax output needs argmax
        pred_ids = np.argmax(predictions[0], axis=-1)
    
    # Convert to tags
    idx2tag = {v: k for k, v in tag2idx.items()}
    pred_tags = [idx2tag.get(idx, 'O') for idx in pred_ids[:len(tokens)]]
    
    # Create result
    result = list(zip(tokens, pred_tags))
    return result

def create_prediction_service():
    """Create a simple prediction service"""
    print("🔧 Loading model for prediction service...")
    
    try:
        # Load model and vocabularies
        model = load_trained_model('best_model.keras')
        
        with open('word2idx.json', 'r') as f:
            word2idx = json.load(f)
        with open('tag2idx.json', 'r') as f:
            tag2idx = json.load(f)
        with open('model_config.json', 'r') as f:
            model_config = json.load(f)
        
        char2idx = None
        if model_config.get('use_char_features', False):
            try:
                with open('char2idx.json', 'r') as f:
                    char2idx = json.load(f)
            except FileNotFoundError:
                print("⚠️  Character vocabulary not found")
        
        print("✅ Model and vocabularies loaded successfully")
        
        # Create a simple config object
        class PredConfig:
            def __init__(self, config_dict):
                for k, v in config_dict.items():
                    setattr(self, k.upper(), v)
        
        pred_config = PredConfig(model_config)
        
        def predict_text(text):
            """Predict entities in text"""
            return predict_entities(model, text, word2idx, char2idx, tag2idx, pred_config)
        
        return predict_text
        
    except Exception as e:
        print(f"❌ Failed to create prediction service: {e}")
        return None

# --------------------------
# Run Training
# --------------------------
if __name__ == "__main__":
    try:
        # Run main training
        model, word2idx, tag2idx, char2idx, history = main()
        
        # Optionally create prediction service
        print("\n🔧 Creating prediction service...")
        predict_func = create_prediction_service()
        
        if predict_func:
            # Test prediction
            test_sentence = "Dr. John Smith works at Stanford University"
            result = predict_func(test_sentence)
            print(f"\n🧪 Test prediction:")
            print(f"   Input: {test_sentence}")
            print(f"   Output: {result}")
        
        print("\n✨ All done! Your enhanced NER model is ready for use.")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()