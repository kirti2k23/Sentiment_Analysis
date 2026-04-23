# Social Media Sentiment Analysis

A multi-model NLP pipeline that classifies social media posts (Twitter, Facebook, Instagram) as **positive**, **negative**, or **neutral** using four text encoding strategies and six machine learning classifiers — 24 model combinations in total.

---

## Dataset

| Field | Detail |
|---|---|
| Source | `sentiment_analysis.csv` (Kaggle) |
| Rows | 499 |
| Columns | Year, Month, Day, Time of Tweet, text, sentiment, Platform |
| Target | `sentiment` — positive / negative / neutral |
| Platforms | Twitter, Facebook, Instagram |

---

## Pipeline

```
Raw Text
   │
   ▼
1. EDA ──────────────────── sentiment distribution, platform/time breakdowns,
   │                         feature correlation, platform × time heatmap
   ▼
2. Text Preprocessing ────── lowercase → strip URLs/mentions/hashtags →
   │                         remove punctuation & digits → tokenize →
   │                         remove stopwords → lemmatize (NLTK)
   ▼
3. Text Encoding (4 methods)
   ├── BOW             (499 × 1228) — sparse count matrix
   ├── TF-IDF          (499 × 3000) — weighted unigrams + bigrams
   ├── Word2Vec        (499 × 100)  — averaged Gensim embeddings
   └── Transformer     (499 × 768)  — DistilBERT sentence embeddings
   │
   ▼
4. Classification (6 models × 4 embeddings = 24 runs)
   ├── Logistic Regression
   ├── SVM (Linear)
   ├── Random Forest
   ├── Gradient Boosting
   ├── KNN (cosine)
   └── Naive Bayes (Multinomial for sparse / Gaussian for dense)
   │
   ▼
5. Evaluation ───────────── Accuracy · F1-Weighted · F1-Macro
                             confusion matrix for best model
```

---

## Results

### Full Leaderboard (sorted by Accuracy)

| Embedding | Classifier | Accuracy | F1 Weighted | F1 Macro |
|---|---|---|---|---|
| **Transformer** | **SVM (Linear)** | **79.0%** | **78.8%** | **78.6%** |
| Transformer | Logistic Regression | 78.0% | 77.7% | 77.4% |
| BOW | SVM (Linear) | 78.0% | 77.2% | 76.2% |
| Transformer | Random Forest | 76.0% | 75.8% | 75.4% |
| BOW | Logistic Regression | 75.0% | 74.0% | 72.8% |
| TF-IDF | Naive Bayes | 73.0% | 71.9% | 70.8% |
| TF-IDF | SVM (Linear) | 72.0% | 70.5% | 69.2% |
| Transformer | KNN | 72.0% | 72.0% | 72.2% |
| BOW | Naive Bayes | 71.0% | 70.7% | 69.9% |
| Word2Vec | Random Forest | 62.0% | 58.1% | 55.2% |
| Word2Vec | Logistic Regression | 40.0% | 22.9% | 19.0% |

### Best Model — Classification Report

**Transformer + SVM (Linear) → 79% Accuracy**

```
              precision  recall  f1-score  support
   negative      0.76    0.70      0.73       27
    neutral      0.79    0.75      0.77       40
   positive      0.81    0.91      0.86       33
   accuracy                        0.79      100
```

---

## Key Findings

**By Embedding Method:**
- **Transformer** — best overall; DistilBERT contextual embeddings give every classifier a strong signal
- **BOW** — strong sparse baseline; SVM and Logistic Regression still hit 78% and 75%
- **TF-IDF** — bigram weighting helps linear models; competitive with BOW
- **Word2Vec** — averaged embeddings lose context; weaker for linear separability

**By Classifier:**
- **SVM (Linear)** — best performer on sparse (BOW/TF-IDF) and dense (Transformer) alike
- **Logistic Regression** — very close to SVM; robust across all embeddings
- **KNN** — cosine similarity works well in the 768-d Transformer space
- **Naive Bayes** — fastest; competitive on TF-IDF, weaker on dense embeddings

**EDA Takeaways:**
- Positive tweets peak in the **morning**; negativity peaks at **night**
- Sentiment has **very low linear correlation** with all temporal/platform features — text content is the critical signal
- **Facebook in the morning** has the highest positive rate; **Instagram at night** the lowest

---

## Visualizations

| File | Contents |
|---|---|
| `sentiment_eda.png` | 9-panel EDA dashboard — distribution, platform/time breakdowns, correlation heatmap |
| `classification_accuracy.png` | Grouped bar chart — accuracy per classifier × embedding |
| `classification_heatmaps.png` | Heatmaps for Accuracy, F1-Weighted, F1-Macro across all 24 combinations |
| `classification_radar.png` | Radar chart — mean performance per embedding method |
| `best_model_confusion_matrix.png` | Confusion matrix for Transformer + SVM |

---

## Project Structure

```
Kaggle_Projects/
├── Sentiment_analysis.ipynb          # Full pipeline notebook
├── sentiment_analysis.csv            # Raw dataset
├── requirements.txt                  # Python dependencies
├── sentiment_eda.png
├── classification_accuracy.png
├── classification_heatmaps.png
├── classification_radar.png
└── best_model_confusion_matrix.png
```

---

## Setup & Run

```bash
# 1. Clone / navigate to the project
cd Kaggle_Projects

# 2. Create and activate a virtual environment
python -m venv env
source env/bin/activate      # Windows: env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download required NLTK data (runs automatically on first cell execution)
#    or pre-download with:
python -c "import nltk; nltk.download(['stopwords','wordnet','punkt','punkt_tab'])"

# 5. Open and run the notebook
jupyter notebook Sentiment_analysis.ipynb
```

> **Note:** The Transformer step downloads `distilbert-base-nli-mean-tokens` (~265 MB) on first run via `sentence-transformers`. A GPU is optional but speeds up encoding.

---

## Dependencies

See [requirements.txt](requirements.txt) for pinned versions.

Core libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `nltk`, `gensim`, `sentence-transformers`, `torch`
