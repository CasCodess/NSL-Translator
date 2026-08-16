"""
Step 2: Train the Classifier
-----------------------------
Reads data/landmarks.csv (built by collect_data.py) and trains a
RandomForest classifier to recognise the signs you recorded.

Run:
    python train_model.py
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

DATA_PATH = os.path.join("data", "landmarks.csv")
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


def train():
    if not os.path.isfile(DATA_PATH):
        print(f"No data found at {DATA_PATH}. Run collect_data.py first for each sign.")
        return

    df = pd.read_csv(DATA_PATH)
    if df["label"].nunique() < 2:
        print("You need at least 2 different signs recorded before training.")
        return

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    clf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(clf, os.path.join(MODEL_DIR, "nsl_model.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print(f"Model saved to {MODEL_DIR}/nsl_model.pkl")


if __name__ == "__main__":
    train()
