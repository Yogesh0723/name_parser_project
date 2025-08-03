import tensorflow as tf
import numpy as np
import json
import csv
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and vocab using standard Keras load_model
model = tf.keras.models.load_model("name_entity_model.keras")
with open("word2idx.json") as f:
    word2idx = json.load(f)
with open("tag2idx.json") as f:
    tag2idx = json.load(f)
idx2tag = {v: k for k, v in tag2idx.items()}

# Clean and tokenize name string
def clean_and_tokenize(line: str):
    line = line.strip().replace("\t", " ").replace("|", " ").replace(".", " ")
    line = re.sub(r"\s+", " ", line)
    return line.strip().split()

# Encode tokens into IDs
def encode_tokens(tokens):
    return [word2idx.get(tok, word2idx["UNK"]) for tok in tokens]

# Predict and organize labels
def predict_name(name_input: str):
    tokens = clean_and_tokenize(name_input)
    input_ids = encode_tokens(tokens)
    # Use a fixed maxlen based on the training data (or adjust as needed)
    maxlen = 30  # Adjust this value based on your model's expected input length
    padded = pad_sequences([input_ids], maxlen=maxlen, padding='post')
    # Standard Keras model prediction
    pred_probs = model.predict(tf.constant(padded), verbose=0)[0]
    pred_ids = np.argmax(pred_probs, axis=-1)

    result = {
        "Input": name_input,
        "B-SAL": "",
        "B-FOR": "",
        "B-MID": "",
        "B-SUR": ""
    }

    for i, idx in enumerate(pred_ids):
        if i >= len(tokens):
            break
        tag = idx2tag[idx]
        if tag in result:
            result[tag] = tokens[i]
    return result

# Main runner
def process_file(input_file="names.txt", output_file="predicted_names.csv"):
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
    process_file("names.txt", "predicted_names.csv")
