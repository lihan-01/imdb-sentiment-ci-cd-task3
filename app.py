import os
import joblib
import gradio as gr


MODEL_PATH = "model.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "model.joblib not found. Please run train.py first."
    )

model = joblib.load(MODEL_PATH)


def predict_sentiment(review):
    if not review or not review.strip():
        return "Please enter a movie review.", 0.0

    prediction = model.predict([review])[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba([review])[0]
        confidence = float(max(probability))
    else:
        confidence = 0.0

    label = "Positive" if prediction == 1 else "Negative"

    return label, round(confidence, 4)


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=8,
        label="IMDB Movie Review",
        placeholder="Type a movie review here..."
    ),
    outputs=[
        gr.Label(label="Predicted Sentiment"),
        gr.Number(label="Confidence"),
    ],
    title="IMDB Sentiment Analysis with Neural Network",
    description=(
        "This app uses TF-IDF features and an MLP neural network "
        "to classify IMDB movie reviews as positive or negative."
    ),
    examples=[
        ["This movie was amazing. The story was emotional and the acting was excellent."],
        ["The film was boring, too long, and the characters were badly written."],
    ],
)


if __name__ == "__main__":
    demo.launch()