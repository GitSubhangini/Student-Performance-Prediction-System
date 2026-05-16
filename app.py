# ============================================================
#  app.py  —  Flask Web Application
#  PURPOSE: Serve a web UI where the user enters reading &
#           writing scores and receives a predicted math score.
#  RUN:     python app.py
#  OPEN:    http://127.0.0.1:5000
# ============================================================

# ── Imports ────────────────────────────────────────────────
# Standard library
import os

# Third-party libraries
import numpy as np          # numerical operations (np.clip)
import pandas as pd         # DataFrame — keeps column names so sklearn won't warn
import joblib               # load the saved model from disk

# Flask — web framework
from flask import Flask, render_template, request, send_from_directory

# ── App Setup ──────────────────────────────────────────────
# __name__ tells Flask where to look for templates/ and static/
app = Flask(__name__)

# ── Load Saved Model ───────────────────────────────────────
# IMPORTANT: Run train_model.py FIRST to create model/model.pkl
MODEL_PATH = os.path.join("model", "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "[ERROR] model/model.pkl not found!\n"
        "   Please run:  python train_model.py  first."
    )

model = joblib.load(MODEL_PATH)
print("[OK] Model loaded successfully!")


# ── Grade Helper ───────────────────────────────────────────
def get_grade(score):
    """Return a grade letter, description, and Bootstrap colour class."""
    score = float(score)
    if score >= 90:
        return "A+", "Outstanding! Keep it up!",  "success"
    elif score >= 80:
        return "A",  "Excellent Work!",            "success"
    elif score >= 70:
        return "B",  "Good Work! Keep going!",      "info"
    elif score >= 60:
        return "C",  "Average — study more!",       "warning"
    elif score >= 50:
        return "D",  "Below Average — work harder", "warning"
    else:
        return "F",  "Needs Improvement — don't give up!", "danger"


# ══════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════

@app.route("/favicon.ico")
def favicon():
    """
    Serve a favicon so browsers don't generate a 404 on every page load.
    We serve a tiny blank icon from the static folder.
    """
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon"
    )


@app.route("/")
def home():
    """
    Home page route.
    Renders the prediction form.
    Flask looks for templates/index.html automatically.
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction route — triggered when the user submits the form.
    Steps:
      1. Read form values (reading_score, writing_score)
      2. Validate them
      3. Pass to the ML model
      4. Return the result page
    """
    error = None
    prediction = None
    grade = label = badge = None
    reading = writing = ""

    try:
        # Read form values sent by the HTML form
        reading = request.form.get("reading_score", "").strip()
        writing = request.form.get("writing_score", "").strip()

        # Basic validation
        if not reading or not writing:
            error = "Please enter both Reading Score and Writing Score."
        else:
            reading_val = float(reading)
            writing_val = float(writing)

            # Validate score range
            if not (0 <= reading_val <= 100 and 0 <= writing_val <= 100):
                error = "Scores must be between 0 and 100."
            else:
                # Build a DataFrame with the EXACT column names used during training.
                # This eliminates the sklearn "feature names" UserWarning.
                features = pd.DataFrame(
                    [[reading_val, writing_val]],
                    columns=["reading score", "writing score"]
                )

                # Run the prediction — no warnings now!
                predicted_math = model.predict(features)[0]

                # Clamp between 0 and 100
                prediction = round(float(np.clip(predicted_math, 0, 100)), 2)

                # Get grade info
                grade, label, badge = get_grade(prediction)

    except ValueError:
        error = "Please enter valid numbers for the scores."
    except Exception as exc:
        error = f"An error occurred: {str(exc)}"

    return render_template(
        "index.html",
        prediction = prediction,
        grade      = grade,
        label      = label,
        badge      = badge,
        reading    = reading,
        writing    = writing,
        error      = error,
    )


# ── Run ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Student Performance Prediction System")
    print("  URL: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    # debug=True → auto-reloads when you save changes (great for learning!)
    app.run(host="0.0.0.0", port=5000)
