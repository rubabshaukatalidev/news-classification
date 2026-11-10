# 📰 Urdu News Classifier

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A full-stack **Urdu news topic classification** app: a FastAPI backend serving a TF-IDF + Logistic Regression
model, and a clean right-to-left web frontend. Built for the **Neural Minds Community of Pakistan —
Code for Intelligence AI Hackathon, challenge 4.3 (Digital Media: Urdu News Topic Classification)**.

> اردو خبر لکھیں، موضوع فوراً معلوم کریں — *Paste an Urdu headline, get its topic instantly.*

## Features

- **6 topics:** Politics (سیاست), Health (صحت), Education (تعلیم), Economy (معیشت), Sports (کھیل), Technology (ٹیکنالوجی)
- **Single & batch** classification (up to 100 headlines per request)
- **Confidence scores** for every class, shown as bars in the UI
- **Urdu-aware preprocessing** — diacritic/variant-letter normalisation; word + character n-gram TF-IDF
- **REST API** with auto-generated docs at `/docs`
- **Retrain on your own data** (CSV or XLSX) with one command
- Dark-mode friendly, responsive, no frontend build step
- Dockerfile and test suite included

## Quick start

```bash
git clone https://github.com/<your-username>/urdu-news-classifier.git
cd urdu-news-classifier

python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m training.train                               # trains the model (takes seconds)
uvicorn backend.app:app --reload                       # start the server
```

Open **http://localhost:8000** for the UI and **http://localhost:8000/docs** for the API docs.
If you skip the training step, the server trains on the bundled seed data automatically on first start.

### Docker

```bash
docker build -t urdu-news-classifier .
docker run -p 8000:8000 urdu-news-classifier
```

## API

| Method | Endpoint              | Description                                  |
|--------|-----------------------|----------------------------------------------|
| GET    | `/api/health`         | Liveness check + list of classes             |
| GET    | `/api/categories`     | Category labels in English and Urdu          |
| GET    | `/api/metrics`        | Training summary (samples, CV macro-F1)      |
| POST   | `/api/classify`       | Classify one text                            |
| POST   | `/api/classify/batch` | Classify up to 100 texts                     |

```bash
curl -X POST http://localhost:8000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "پاکستان کرکٹ ٹیم نے سیریز کا پہلا میچ جیت لیا"}'
```

```json
{
  "label": "sports",
  "label_ur": "کھیل",
  "label_en": "Sports",
  "confidence": 0.9395,
  "scores": [{"label": "sports", "label_ur": "کھیل", "label_en": "Sports", "score": 0.9395}, "..."]
}
```

## Project structure

```
urdu-news-classifier/
├── backend/app.py          # FastAPI app + static frontend hosting
├── training/
│   ├── train.py            # training + evaluation script
│   └── text.py             # Urdu normalisation
├── frontend/index.html     # single-file RTL web UI
├── data/seed_urdu_news.csv # small bundled demo dataset (120 headlines)
├── models/                 # trained model + metrics (generated)
├── tests/test_api.py       # API tests (pytest)
├── Dockerfile
└── requirements.txt
```

## ⚠️ Important: about the bundled data and accuracy

The model that ships here is trained on **`data/seed_urdu_news.csv` — 120 short, hand-written Urdu
headlines (20 per topic)**. This is a **demo dataset to make the project run out of the box**, not a
real-world benchmark.

- Measured 5-fold cross-validated macro-F1 on this seed set is about **0.74**. Expect weaker, less
  stable results on real articles, especially long text or topics outside the six classes.
- The hackathon's recommended corpus ([Data for Urdu Fake News, Zenodo](https://zenodo.org/records/7773474))
  carries **authenticity labels, not topic labels**. As the challenge brief notes, topic labels must be
  created and quality-checked by the organisers/participants under a clear taxonomy. Do not treat
  fake/real labels as topics.
- For a serious submission, annotate or collect a larger topic-labelled set (hundreds+ per class) and retrain.

### Retrain with your own data

```bash
python -m training.train --data data/external/my_urdu_news.csv
python -m training.train --data news.xlsx --text-col headline --label-col topic
```

The file needs one text column and one label column. Labels matching the six built-in keys
(`politics, health, education, economy, sports, technology`) get Urdu/English display names; any new
label still works and is shown with its raw name. The backend automatically picks up whatever classes
the model was trained on.

## Responsible use

- Predictions are **automated suggestions**; a human editor should review before publishing or archiving.
- This is a *topic* classifier. It does **not** judge whether a story is true or false.
- Remove personal information from any data you add, and respect each dataset's licence.

## Development

```bash
pip install -r requirements.txt
python -m training.train
pytest
```

## Roadmap

- [ ] Larger topic-labelled Urdu corpus + hidden test split
- [ ] Multilingual transformer baseline (e.g. mBERT / XLM-R) for comparison
- [ ] Roman Urdu support
- [ ] Per-topic confusion matrix in the UI

## Acknowledgements

Challenge concept and dataset guidance from the **Neural Minds Community of Pakistan —
Code for Intelligence AI Hackathon** proposal.

## License

[MIT](LICENSE)
