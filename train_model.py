"""
train_model.py
--------------
Trains a Linear SVC classifier on the FULL product dataset and saves
the model and vectorizer for later use in predict_category.py.

Usage:
    python train_model.py
"""

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH       = "data/products.csv"
MODEL_PATH      = "models/category_classifier.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

# ── Load data ───────────────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH, encoding="latin-1")
print(f"   Loaded {df.shape[0]:,} products")

# ── Clean ───────────────────────────────────────────────────────────────────
print("🧹 Cleaning data...")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Product Title", "Category Label"])

df["Category Label"] = df["Category Label"].replace({
    "fridge":       "Fridges",
    "CPU":          "CPUs",
    "Mobile Phone": "Mobile Phones"
})

# Clean: Keep only relevant columns
df = df[["Product Title", "Category Label"]]
print(f"   Clean dataset: {df.shape[0]:,} products, {df['Category Label'].nunique()} categories")

# ── Prepare features ────────────────────────────────────────────────────────
X = df["Product Title"]
y = df["Category Label"]

# ── Vectorize ───────────────────────────────────────────────────────────────
print("🔢 Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)
print(f" Vocabulary size: {len(vectorizer.vocabulary_):,} words")

# ── Train on full dataset ───────────────────────────────────────────────────
print("🤖 Training Linear SVC on full dataset...")
model = LinearSVC(random_state=42)
model.fit(X_tfidf, y)
print(" Training complete!")

# ── Save ────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print(f"\n✅ Model saved     → {MODEL_PATH}")
print(f"✅ Vectorizer saved → {VECTORIZER_PATH}")
print("\n🎯 Ready for predictions! Run: python predict_category.py")
