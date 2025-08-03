import tensorflow as tf
import numpy as np
import json
import csv
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import unicodedata

# Load model
model = tf.keras.models.load_model("name_entity_functional_model.keras")

# Load vocab mappings
with open("word2idx.json") as f:
    word2idx = json.load(f)
with open("tag2idx.json") as f:
    tag2idx = json.load(f)
idx2tag = {v: k for k, v in tag2idx.items()}

MAX_LEN = 30  # Must match training

# 🔄 Use same normalization as in training
def normalize_token(token):
    return unicodedata.normalize("NFKC", token).strip()

def clean_and_tokenize(line: str):
    line = unicodedata.normalize("NFKC", line)  # normalize full string
    line = re.sub(r"[|.]", " ", line)           # replace unwanted chars
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

    # Organize results — support multiple tokens per tag
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

    # Join tokens back to string
    return {
        "Input": name_input,
        "B-SAL": " ".join(result["B-SAL"]),
        "B-FOR": " ".join(result["B-FOR"]),
        "B-MID": " ".join(result["B-MID"]),
        "B-SUR": " ".join(result["B-SUR"]),
    }


# Read names from file and write predictions to CSV
def process_file(input_file="names.txt", output_file="predicted_names_functional_2.csv"):
    output_rows = []
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if line.strip():
            prediction = predict_name(line.strip())
            output_rows.append(prediction)

    # Write to CSV
    with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = ["Input", "B-SAL", "B-FOR", "B-MID", "B-SUR"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"✅ Predictions saved to: {output_file}")

if __name__ == "__main__":
    process_file("names.txt", "predicted_names_functional_2.csv")
