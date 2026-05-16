# AI Student Analytics System

Advanced local Flask app for student final-score prediction, risk detection, explanations, dashboards, CSV upload, and reports.

## Run

```bash
pip install -r requirements.txt
python generate_data.py
python src/pipeline/train_pipeline.py
python app.py
```

Open `http://127.0.0.1:5000`.

Development admin login is `admin / admin123` unless you set `ADMIN_USERNAME`, `ADMIN_EMAIL`, and `ADMIN_PASSWORD`.

## Test

```bash
python -m unittest discover -s tests
```
