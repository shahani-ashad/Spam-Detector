import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# load dataset
data = pd.read_csv("final.csv")

# normalize column names
data.columns = data.columns.str.strip().str.lower()

# clean text column
data = data.dropna(subset=["text"])
data["text"] = data["text"].astype(str)
data = data[data["text"].str.strip() != ""]

# features & labels
X = data["text"]
y = data["label"]

# split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# pipeline: text → numbers → model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", RandomForestClassifier())
])

# train
model.fit(x_train, y_train)

# save model
pickle.dump(model, open("model.pkl", "wb"))

print("Training complete")
