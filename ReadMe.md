# Fake News Classification Bot — MLP

A full-stack web application that classifies news articles as **real** or **fake** using a Multi-Layer Perceptron (MLP) trained on labeled news data. The system exposes a Flask REST API backend and a static HTML/CSS/JS frontend.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Contributing](#contributing)

---

## Overview

This project addresses the problem of misinformation by providing an automated fake news classifier. A user submits a news article (headline or body text) through the frontend UI, the text is sent to the backend, preprocessed, and passed through a trained MLP model that returns a real/fake prediction.

---

## Architecture

```
┌────────────────────────┐         HTTP Request         ┌──────────────────────────┐
│                        │ ─────────────────────────►  │                          │
│   Frontend (Static)    │                              │   Flask Backend (API)    │
│   HTML / CSS / JS      │ ◄─────────────────────────  │   app.py                 │
│   Port 5500 / 5501     │         JSON Response        │   Port 5000              │
└────────────────────────┘                              └──────────┬───────────────┘
                                                                   │
                                                          ┌────────▼────────┐
                                                          │   ML Pipeline   │
                                                          │  TF-IDF + MLP   │
                                                          │  scikit-learn   │
                                                          └─────────────────┘
```

The frontend communicates with the backend over HTTP. The backend preprocesses incoming text, vectorizes it using TF-IDF, and runs inference through the trained MLP classifier.

---

## Project Structure

```
fake_news_classificactionbot_MLP/
├── backend/
│   ├── venv/                  # Python virtual environment (not committed)
│   ├── app.py                 # Flask application entry point
│   ├── model.pkl              # Serialized trained MLP model
│   ├── vectorizer.pkl         # Serialized TF-IDF vectorizer
│   └── requirements.txt       # Python dependencies
├── data/
│   ├── Fake.csv               # Labeled fake news samples
│   └── True.csv               # Labeled real news samples
├── frontend/
│   ├── index.html             # Main UI
│   ├── style.css              # Stylesheet
│   └── script.js              # API call logic
├── .gitignore
└── ReadMe.md
```

---

## Tech Stack

| Layer      | Technology                              |
|------------|-----------------------------------------|
| Frontend   | HTML5, CSS3, Vanilla JavaScript         |
| Backend    | Python 3, Flask                         |
| ML Model   | scikit-learn (MLP Classifier, TF-IDF)   |
| Data       | CSV (Fake.csv / True.csv)               |
| Packaging  | Python venv                             |

---

## Prerequisites

- Python **3.8+**
- pip
- A terminal with `python3` and `venv` available

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/intxred/fake_news_classificactionbot_MLP.git
cd fake_news_classificactionbot_MLP
```

### 2. Set up the Python virtual environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

### Start the backend

```bash
cd backend
source venv/bin/activate
python3 app.py
```

The Flask server starts on **http://localhost:5000** by default.

### Start the frontend

Open a new terminal from the project root and serve the static frontend:

```bash
python3 -m http.server 5500
```

If port 5500 is already in use:

```bash
python3 -m http.server 5501
```

Then open your browser at **http://localhost:5500** (or 5501).

---

## API Reference

### `POST /predict`

Accepts a news text string and returns a classification.

**Request**

```http
POST /predict
Content-Type: application/json

{
  "text": "Scientists discover cure for all diseases overnight."
}
```

**Response**

```json
{
  "prediction": "FAKE",
  "confidence": 0.93
}
```

| Field        | Type   | Description                                  |
|--------------|--------|----------------------------------------------|
| `prediction` | string | `"REAL"` or `"FAKE"`                         |
| `confidence` | float  | Model confidence score between 0.0 and 1.0   |

---

## Model Details

The classifier is a **Multi-Layer Perceptron (MLP)** trained with scikit-learn.

**Pipeline:**

1. **Text preprocessing** — lowercasing, punctuation removal, stopword filtering
2. **TF-IDF Vectorization** — converts cleaned text into a numerical feature matrix
3. **MLP Classifier** — a feedforward neural network with one or more hidden layers trained to distinguish real from fake articles
4. **Serialization** — the trained vectorizer and model are saved as `vectorizer.pkl` and `model.pkl` using `joblib`/`pickle` for fast inference at runtime

**Metrics evaluated during training:**
- Accuracy
- Precision
- Recall
- F1-score

---

## Dataset

The project uses two CSV files located in the `/data` directory:

| File       | Content                              |
|------------|--------------------------------------|
| `Fake.csv` | Labeled fake news articles           |
| `True.csv` | Labeled real/legitimate news articles|

Both files are combined, shuffled, and split into training and test sets before model training. The combined dataset provides balanced class representation for binary classification.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

This project does not currently specify a license. Please contact the repository owner before using it in production or redistributing.
