from flask import Flask, request, render_template
import pickle
import re

app = Flask(__name__)

# Load model and label encoder
model, le = pickle.load(open("model1.pkl", "rb"))

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\u0D80-\u0DFF\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "")

    if text.strip() == "":
        return render_template("index.html",
                               prediction_text="Please enter a comment!")

    cleaned = clean_text(text)

    prediction_index = model.predict([cleaned])[0]
    prediction_label = le.inverse_transform([prediction_index])[0]

    return render_template(
        "index.html",
        prediction_text=f"Prediction: {prediction_label}"
    )

if __name__ == "__main__":
    app.run(debug=True)
