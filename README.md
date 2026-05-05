# 🛍️ Product Category Classifier

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-3.x-11557C?logo=matplotlib&logoColor=white)
![seaborn](https://img.shields.io/badge/seaborn-0.13-4C72B0)
![joblib](https://img.shields.io/badge/joblib-1.x-FF6B6B)

An end-to-end machine learning project for automatic product category prediction
based on product titles — built for an e-commerce platform handling 35,000+ products.

Given a product title like `"bosch serie 4 kgv39vl31g"`, the model automatically
predicts the correct category — in this case `Fridge Freezers`.

---

## 📁 Project structure

```
product-category-classifier/
├── data/
│   ├── products.csv                            # Raw dataset (35,311 products)
│   ├── exploration.png                       # Category distribution charts
│   ├── exploration_clean.png            # Category distribution charts
│   ├── missing_after.png                   # Category distribution charts
│   └── confusion_matrices.png          # Model evaluation visualizations
├── models/
│   ├── category_classifier.pkl        # Trained Linear SVC model (full dataset)
│   └── vectorizer.pkl                        # Fitted TF-IDF vectorizer
├── notebooks/
│   └── product_category_analysis.ipynb  # Full analysis and model development
├── train_model.py                                    # Trains and saves the model
├── predict_category.py                           # Interactive prediction script
├── requirements.txt                                # Project dependencies
├── .gitignore
└── README.md
```

---

## 🤖 Models compared

Four classification algorithms were evaluated on an 80/20 train/test split:

| Model | Accuracy | Macro F1 | Weakest category |
|-------|----------|----------|-----------------|
| Logistic Regression | 96.00% | 0.96 | Fridges (0.90) |
| Multinomial Naive Bayes | 93.96% | 0.93 | Freezers (0.70) ⚠️ |
| Decision Tree | 93.97% | 0.94 | Fridges (0.87) |
| **Linear SVC** ✅ | **97.21%** | **0.97** | Fridges (0.93) |

> The final model (`category_classifier.pkl`) is trained on the **full dataset**
> for maximum performance in production.

---

## 🏆 Result

**Linear SVC** was selected as the best model with **97.21% accuracy** and
macro F1 score of **0.97** — consistent performance across all 10 product categories.

The biggest challenge for all models was distinguishing between similar categories:
`Fridges`, `Freezers`, and `Fridge Freezers` — products with similar titles
that are genuinely difficult to tell apart from the name alone.

---

## 🚀 How to run

### 1. Clone the repository
```bash
git clone https://github.com/ivanadjuricic/product-category-classifier.git
cd product-category-classifier
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train_model.py
```

### 5. Predict categories interactively
```bash
python predict_category.py
```

### 6. Open the notebook (optional)
```bash
# Register the kernel first
python -m ipykernel install --user --name=venv --display-name "Python (venv)"

# Open notebook
jupyter notebook notebooks/product_category_analysis.ipynb
```

---

## 🧪 Sample predictions

| Product title | Expected category |
|---------------|------------------|
| Apple iPhone 7 32GB | Mobile Phones |
| olympus e m10 mark iii geh use silber | Digital Cameras |
| kenwood k20mss15 solo | Microwaves |
| bosch wap28390gb 8kg 1400 spin | Washing Machines |
| bosch serie 4 kgv39vl31g | Fridge Freezers |
| smeg sbs8004po | Fridge Freezers |

---

## 📦 Dependencies

```
pandas
scikit-learn
ipykernel
matplotlib
seaborn
notebook
joblib
```

---

## ⚠️ Model limitations

During testing, two out of six sample products were misclassified:

| Product title | Predicted | Expected |
|---------------|-----------|----------|
| bosch serie 4 kgv39vl31g | Dishwashers | Fridge Freezers |
| smeg sbs8004po | Dishwashers | Fridge Freezers |

Both titles consist mostly of **product codes with no descriptive keywords** —
the model has no textual signal to work with. This is a known limitation:
when a product title is too short or too technical, even a human would struggle
to identify the correct category without additional information.

A potential improvement would be to enrich short titles with additional data
such as product descriptions or specifications before classification.

*Course Assignment: Prediction of Product Category from Product Title*
