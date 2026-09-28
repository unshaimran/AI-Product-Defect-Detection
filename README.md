# 🔍 AI Product Defect Detection System

An AI-powered product defect detection application built with Python and Streamlit. The application analyzes uploaded product images using a pre-trained CLIP vision model and identifies possible product conditions.

## 🚀 Features

- 📷 Upload product images
- 🤖 AI-powered image classification
- 🔍 Detect possible scratches, cracks, damage, or defects
- 📊 Display prediction confidence
- 📈 Show alternative predictions
- 📝 Generate an analysis summary
- 📄 Download a defect analysis report
- 🎨 Interactive Streamlit interface

## 🛠️ Technologies Used

- Python
- Streamlit
- Hugging Face Transformers
- CLIP
- PyTorch
- PIL

## 🧠 AI Model

This project uses the pre-trained:

`openai/clip-vit-base-patch32`

model through Hugging Face Transformers for zero-shot image classification.

## 📂 Project Structure

```text
AI-Product-Defect-Detection/
│
├── app.py
├── requirements.txt
└── .gitignore
