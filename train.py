import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


DATA_PATH = "data/imdb_balanced_10k.csv"
MODEL_PATH = "model.joblib"


def find_text_column(df):
    candidates = ["review", "text", "comment", "content", "description"]

    for col in candidates:
        if col in df.columns:
            return col

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if not object_cols:
        raise ValueError("No text column found.")

    return max(object_cols, key=lambda c: df[c].astype(str).str.len().mean())


def find_label_column(df, text_col):
    candidates = ["sentiment", "label", "target", "polarity", "class"]

    for col in candidates:
        if col in df.columns and col != text_col:
            return col

    for col in df.columns:
        if col != text_col and df[col].nunique() <= 5:
            return col

    raise ValueError("No label column found.")


def normalize_label(value):
    value = str(value).strip().lower()

    if value in ["positive", "pos", "1", "true", "good"]:
        return 1

    if value in ["negative", "neg", "0", "false", "bad"]:
        return 0

    try:
        number = float(value)
        return 1 if number >= 7 else 0
    except ValueError:
        raise ValueError(f"Unknown label value: {value}")


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Please put imdb_balanced_10k.csv into the data folder."
        )

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Columns:", df.columns.tolist())

    text_col = find_text_column(df)
    label_col = find_label_column(df, text_col)

    print("Text column:", text_col)
    print("Label column:", label_col)

    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].apply(normalize_label)

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=0.2,
        random_state=42,
        stratify=df[label_col],
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    max_iter=20,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training neural network...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", accuracy)
    print(classification_report(y_test, predictions))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()