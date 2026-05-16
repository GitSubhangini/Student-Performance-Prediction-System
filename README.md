# Student Performance Prediction System

This repository contains two versions:

- `AI-Student-Analytics-System/` is the canonical advanced app.
- The root-level Flask files are kept as a small legacy/beginner demo.

## Advanced App Quick Start

```bash
cd AI-Student-Analytics-System
pip install -r requirements.txt
python generate_data.py
python src/pipeline/train_pipeline.py
python app.py
```

Open `http://127.0.0.1:5000`.

Development login defaults to `admin / admin123`. For a safer local setup, set:

```bash
set SECRET_KEY=change-me
set ADMIN_USERNAME=admin
set ADMIN_EMAIL=admin@example.com
set ADMIN_PASSWORD=choose-a-password
python app.py
```

## What The Advanced App Includes

- Leakage-free synthetic data with `final_score` as the target.
- Multi-model training with metrics, model metadata, feature schema, and feature importance.
- Authenticated dashboard, admin panel, prediction form, CSV upload, reports, CSV export, and PDF export.
- Prediction outputs with grade, risk level, confidence, prediction interval, top drivers, recommendations, and model metadata.
- JSON API at `/api/predict`.
- Focused tests covering validation, training artifacts, prediction, auth, dashboard, API, CSV upload, and exports.

## Tests

```bash
cd AI-Student-Analytics-System
python -m unittest discover -s tests
```

## Notes

The project uses only local synthetic data and local ML models. No external AI service or cloud dependency is required.
