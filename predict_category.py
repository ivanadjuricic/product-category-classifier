"""
predict_category.py
-------------------
Loads the trained Linear SVC model and vectorizer, then interactively
predicts the category of a product based on its title.

Usage:
    python predict_category.py
"""

import joblib
import sys

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH      = "models/category_classifier.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

# ── Load model and vectorizer ───────────────────────────────────────────────
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("❌ Model not found. Please run train_model.py first.")
    sys.exit(1)

# ── Test products from the task ─────────────────────────────────────────────
TEST_PRODUCTS = [
    ("Apple iPhone 7 32GB",                    "Mobile Phones"),
    ("olympus e m10 mark iii geh use silber",   "Digital Cameras"),
    ("kenwood k20mss15 solo",                   "Microwaves"),
    ("bosch wap28390gb 8kg 1400 spin",          "Washing Machines"),
    ("bosch serie 4 kgv39vl31g",               "Fridge Freezers"),
    ("smeg sbs8004po",                          "Fridge Freezers"),
]

print("\n── Quick test with sample products ────────────────────────────────")
print(f"  {'Product Title':<45} {'Predicted':<20} {'Expected'}")
print(f"  {'─'*45} {'─'*20} {'─'*20}")

for title, expected in TEST_PRODUCTS:
    vec  = vectorizer.transform([title])
    pred = model.predict(vec)[0]
    match = "✅" if pred == expected else "❌"
    print(f"  {title:<45} {pred:<20} {match}")

# ── Interactive mode ────────────────────────────────────────────────────────
print("\n── Interactive mode (type 'quit' to exit) ──────────────────────────")

while True:
    title = input("\nEnter product title: ").strip()

    if title.lower() in ("quit", "exit", "q"):
        print("👋 Goodbye!")
        break

    if not title:
        print("⚠️  Please enter a product title.")
        continue

    vec  = vectorizer.transform([title])
    pred = model.predict(vec)[0]
    print(f"🏷️  Predicted category: {pred}")
