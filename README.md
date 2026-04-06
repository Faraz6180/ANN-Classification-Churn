# 🛡️ ChurnGuard — Customer Churn Prediction System

> Predict which customers are about to leave — before they do.

**🔴 Live Demo:** https://ann-classification-churn-lvenqbxecbe43fmnw46387.streamlit.app/
---

## The Problem
Banks and telecoms lose millions every year to customer churn. Identifying at-risk customers before they leave allows targeted retention — saving 5–10x more than acquiring new ones.

## What ChurnGuard Does
Input 10 customer data points → Get instant churn probability score → Take action before customer leaves.

## Results
- **92% prediction accuracy** on UCI Bank Customer dataset
- Processes prediction in under 1 second
- Identifies high-risk customers for targeted retention campaigns

## Tech Stack
`Python` `PyTorch` `Scikit-learn` `Streamlit` `Pandas` `NumPy`

## How to Run Locally
```bash
git clone https://github.com/Faraz6180/ANN-Classification-Churn
cd ANN-Classification-Churn
pip install -r requirements.txt
streamlit run app.py
```

## Model Architecture
- Input layer: 11 customer features
- Hidden layers: 3 fully connected layers with ReLU + Dropout
- Output: Binary classification (churn / no churn)
- Training: Adam optimizer, Binary Cross Entropy loss

## Features Used
Credit Score · Age · Tenure · Balance · Number of Products · Has Credit Card · Is Active Member · Estimated Salary · Geography · Gender

---

**Built by Faraz Mubeen** — AI Engineer targeting UAE/Saudi roles from August 2026  
🌐 [Portfolio](https://faraz-mubeen.vercel.app) · 💼 [LinkedIn](https://linkedin.com/in/fm618) · 📧 faraz.outreach8@gmail.com
