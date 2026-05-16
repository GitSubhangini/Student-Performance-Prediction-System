Here's the complete run guide — 3 commands only (after the first-time setup):

🚀 How to Run — AI Student Analytics System
First Time Setup (run once)
Step 1 — Open Terminal in the project folder:

cd "C:\Users\mahap\OneDrive\Documents\Student Performance Prediction System\AI-Student-Analytics-System"
Step 2 — Install dependencies:

pip install flask flask-sqlalchemy flask-login pandas numpy scikit-learn xgboost joblib plotly reportlab
Step 3 — Generate the dataset:

python generate_data.py
Step 4 — Train the ML models:

python src/pipeline/train_pipeline.py
Every Time You Want to Run
Just one command:

python app.py
Then open your browser and go to:

URL	What it is
http://127.0.0.1:5000	Main app (login page)
http://127.0.0.1:5000/dashboard	Analytics Dashboard
http://127.0.0.1:5000/predict	AI Prediction
http://127.0.0.1:5000/report	Reports + PDF
http://127.0.0.1:5000/admin	Admin Panel
Default login credentials:

Username: admin
Password: admin123
To Stop the App
Press Ctrl + C in the terminal.

Quick Visual Summary
student-performance-prediction/
AI-Student-Analytics-System/
│
├── Step 1 → python generate_data.py       ← creates data/student_data.csv
├── Step 2 → python src/pipeline/train_pipeline.py  ← trains model, saves artifacts/
└── Step 3 → python app.py                 ← starts server at localhost:5000
Note: Steps 1–4 (setup) only need to be done once. After that, only python app.py is needed every time.

11:10 PM
