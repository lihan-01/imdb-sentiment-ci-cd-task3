---
title: IMDB Sentiment Analysis
emoji: 🎬
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# IMDB Sentiment Analysis with GitHub CI/CD and Hugging Face

This project is a simple machine learning application for IMDB movie review sentiment analysis.

## Model

The model uses:

- TF-IDF text features
- MLP neural network classifier
- Gradio web interface
- GitHub Actions for CI/CD deployment
- Hugging Face Spaces for hosting

## How It Works

Each IMDB movie review is converted into TF-IDF features.  
The features are passed into an MLP neural network.  
The model predicts whether the review is positive or negative.

## Run Locally

```bash
pip install -r requirements.txt
python train.py
python app.py