from flask import Flask, request, render_template
import pickle
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

# Load LSTM model
model_dl = load_model("model_dl.h5")

# Load LabelEncoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Tokenizer configuration (must match training)
tokenizer = Tokenizer(num_words=40000, oov_token="<OOV>")

# Load training texts to fit tokenizer
import pandas as pd
df = pd.read_csv("Final.csv")

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\u0D80-\u0DFF\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["Text"] = df["Text"].apply(clean_text)
tokenizer.fit_on_texts(df["Text"])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "")
    if text.strip() == "":
        return render_template("index.html", prediction_text="Please enter a comment!")

    cleaned = clean_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=150, padding="post")

    prediction = model_dl.predict(pad)
    prediction_index = np.argmax(prediction, axis=1)[0]
    prediction_label = le.inverse_transform([prediction_index])[0]

    return render_template("index.html", prediction_text=f"Prediction: {prediction_label}")

if __name__ == "__main__":
    app.run(debug=True)

# Save the DL model
model_dl.save("model_dl.pickle")

# Save the LabelEncoder
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("LSTM model and LabelEncoder saved!")
