import tensorflow as tf
import numpy as np
import json
import re
import unicodedata
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and vocab
model = tf.keras.models.load_model("name_entity_functional_model.keras")

with open("word2idx.json") as f:
    word2idx = json.load(f)
with open("tag2idx.json") as f:
    tag2idx = json.load(f)
idx2tag = {v: k for k, v in tag2idx.items()}

MAX_LEN = 30  # Same as training

# Normalization functions
def normalize_token(token):
    return unicodedata.normalize("NFKC", token).strip()

def clean_and_tokenize(line: str):
    line = unicodedata.normalize("NFKC", line)
    line = re.sub(r"[|.]", " ", line)
    line = re.sub(r"\s+", " ", line)
    return [normalize_token(tok) for tok in line.strip().split()]

def encode_tokens(tokens):
    return [word2idx.get(tok, word2idx.get("UNK", 1)) for tok in tokens]

def predict_name(name_input: str):
    tokens = clean_and_tokenize(name_input)
    input_ids = encode_tokens(tokens)
    padded = pad_sequences([input_ids], maxlen=MAX_LEN, padding='post')

    pred_probs = model.predict(tf.constant(padded), verbose=0)[0]
    pred_ids = np.argmax(pred_probs, axis=-1)

    result = {
        "Input": name_input,
        "B-SAL": [],
        "B-FOR": [],
        "B-MID": [],
        "B-SUR": []
    }

    for i, idx in enumerate(pred_ids):
        if i >= len(tokens):
            break
        tag = idx2tag.get(idx, "O")
        if tag in result:
            result[tag].append(tokens[i])

    # Print result nicely
    print("\n🔍 Prediction Result:")
    for key in ["B-SAL", "B-FOR", "B-MID", "B-SUR"]:
        print(f"{key}: {' '.join(result[key]) or '-'}")

# Run interactively
if __name__ == "__main__":
    print("🧠 Name Entity Segmenter (type 'exit' to quit)")
    while True:
        name = input("\nEnter name: ").strip()
        if name.lower() in ["exit", "quit"]:
            break
        if name:
            predict_name(name)
