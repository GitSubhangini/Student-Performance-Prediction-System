"""
AI Student Analytics System - main Flask application.
Run from this folder with: python app.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from functools import wraps

import numpy as np
import pandas as pd
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text
from werkzeug.security import check_password_hash, generate_password_hash

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.helper import get_risk_level
from src.utils.schema import (
    ALL_FEATURES,
    FEATURE_LABELS,
    InputValidationError,
    TARGET,
    missing_csv_columns,
)


DEFAULT_ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@analytics.ai")
DEFAULT_ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
CREATE_DEFAULT_ADMIN = os.environ.get("CREATE_DEFAULT_ADMIN", "1") != "0"


app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "studentai-dev-secret-change-me"),
    SQLALCHEMY_DATABASE_URI=os.environ.get("DATABASE_URL", "sqlite:///analytics.db"),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,
)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "warning"


def utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=utc_now)
    predictions = db.relationship("Prediction", backref="user", lazy=True)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    student_name = db.Column(db.String(100), default="Anonymous")
    predicted_score = db.Column(db.Float)
    grade = db.Column(db.String(5))
    risk_level = db.Column(db.String(10))
    confidence = db.Column(db.Float)
    model_version = db.Column(db.String(80))
    prediction_low = db.Column(db.Float)
    prediction_high = db.Column(db.Float)
    features_json = db.Column(db.Text)
    insights_json = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=utc_now)


@login_manager.user_loader
def load_user(uid):
    try:
        return db.session.get(User, int(uid))
    except (TypeError, ValueError):
        return None


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)

    return decorated


def admin_hint() -> str | None:
    if os.environ.get("SHOW_DEFAULT_ADMIN_HINT", "1") == "0":
        return None
    if os.environ.get("ADMIN_PASSWORD"):
        return f"Configured admin user: {DEFAULT_ADMIN_USERNAME}"
    return f"Development default: {DEFAULT_ADMIN_USERNAME} / admin123"


def load_dataset():
    path = os.path.join("data", "student_data.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def load_metrics():
    path = os.path.join("artifacts", "metrics.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def score_column(df: pd.DataFrame) -> str | None:
    for candidate in [TARGET, "final_score", "average_score"]:
        if candidate in df.columns:
            return candidate
    return None


def dataset_stats(df: pd.DataFrame) -> dict:
    col = score_column(df)
    if df.empty or not col:
        return {}
    risk = df["risk_level"] if "risk_level" in df.columns else df[col].apply(get_risk_level)
    return {
        "total": len(df),
        "avg": round(float(df[col].mean()), 1),
        "high_risk": int((risk == "High").sum()),
        "pass_rate": round(float((df[col] >= 50).mean() * 100), 1),
        "top_scorer": round(float(df[col].max()), 1),
        "score_label": FEATURE_LABELS.get(col, col.replace("_", " ").title()),
    }


def get_predictor():
    """Lazy-load the prediction pipeline."""
    if not hasattr(app, "_predictor"):
        try:
            from src.pipeline.predict_pipeline import PredictPipeline

            app._predictor = PredictPipeline()
        except Exception as e:
            app._predictor = None
            app.logger.warning(f"Predictor not loaded: {e}")
    return app._predictor


def form_payload(form) -> dict:
    return {
        "gender": form.get("gender"),
        "race_ethnicity": form.get("race_ethnicity"),
        "parental_education": form.get("parental_education"),
        "lunch": form.get("lunch"),
        "test_prep_course": form.get("test_prep_course"),
        "extracurricular": form.get("extracurricular"),
        "internet_access": form.get("internet_access"),
        "study_hours_per_week": form.get("study_hours") or form.get("study_hours_per_week"),
        "attendance_rate": form.get("attendance") or form.get("attendance_rate"),
        "math_score": form.get("math_score"),
        "reading_score": form.get("reading_score"),
        "writing_score": form.get("writing_score"),
        "science_score": form.get("science_score"),
    }


def save_prediction(user_id: int, student_name: str, features: dict, result: dict) -> Prediction:
    insights = {
        "drivers": result.get("drivers", []),
        "recommendations": result.get("recommendations", []),
    }
    interval = result.get("interval", {})
    model = result.get("model", {})
    rec = Prediction(
        user_id=user_id,
        student_name=student_name or "Anonymous",
        predicted_score=result["predicted_score"],
        grade=result["grade"],
        risk_level=result["risk_level"],
        confidence=result["confidence"],
        model_version=model.get("version"),
        prediction_low=interval.get("low"),
        prediction_high=interval.get("high"),
        features_json=json.dumps(features),
        insights_json=json.dumps(insights),
    )
    db.session.add(rec)
    return rec


def render_login(**context):
    context.setdefault("default_admin_hint", admin_hint())
    return render_template("login.html", **context)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        user = User.query.filter_by(username=request.form.get("username", "")).first()
        if user and check_password_hash(user.password, request.form.get("password", "")):
            login_user(user, remember=bool(request.form.get("remember")))
            flash(f"Welcome back, {user.username}!", "success")
            return redirect(request.args.get("next") or url_for("dashboard"))
        flash("Invalid username or password.", "danger")
    return render_login()


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
        elif User.query.filter_by(username=username).first():
            flash("Username already taken.", "danger")
        elif User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
        else:
            user = User(
                username=username,
                email=email,
                password=generate_password_hash(password),
                is_admin=User.query.count() == 0,
            )
            db.session.add(user)
            db.session.commit()
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
    return render_login(register=True)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))


@app.route("/")
@app.route("/dashboard")
@login_required
def dashboard():
    df = load_dataset()
    metrics = load_metrics()
    stats = dataset_stats(df)
    recent = (
        Prediction.query.filter_by(user_id=current_user.id)
        .order_by(Prediction.created_at.desc())
        .limit(5)
        .all()
    )
    high_risk = (
        Prediction.query.filter_by(user_id=current_user.id, risk_level="High")
        .order_by(Prediction.created_at.desc())
        .limit(5)
        .all()
    )
    return render_template(
        "dashboard.html",
        stats=stats,
        metrics=metrics,
        recent=recent,
        high_risk=high_risk,
    )


@app.route("/admin")
@login_required
@admin_required
def admin():
    users = User.query.order_by(User.created_at.desc()).all()
    preds = Prediction.query.order_by(Prediction.created_at.desc()).limit(50).all()
    df = load_dataset()
    metrics = load_metrics()
    return render_template(
        "admin.html",
        users=users,
        preds=preds,
        total_students=len(df) if not df.empty else 0,
        total_predictions=Prediction.query.count(),
        admins=User.query.filter_by(is_admin=True).count(),
        model_trained=bool(metrics.get("best_model")),
        best_model=metrics.get("best_model"),
        best_r2=metrics.get("best_r2"),
    )


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict_view():
    result = None
    if request.method == "POST":
        predictor = get_predictor()
        if predictor is None:
            flash("Model not trained yet. Run the training pipeline first.", "danger")
        else:
            try:
                features = form_payload(request.form)
                result = predictor.predict(features)
                save_prediction(
                    current_user.id,
                    request.form.get("student_name", "Anonymous"),
                    features,
                    result,
                )
                db.session.commit()
                flash("Prediction saved with explanation insights.", "success")
            except InputValidationError as e:
                flash(" ".join(e.errors), "danger")
            except Exception as e:
                db.session.rollback()
                flash(f"Prediction error: {e}", "danger")

    leaderboard = Prediction.query.order_by(Prediction.predicted_score.desc()).limit(10).all()
    return render_template("prediction.html", result=result, leaderboard=leaderboard)


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_csv():
    results = []
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename.endswith(".csv"):
            flash("Please upload a valid CSV file.", "danger")
        else:
            predictor = get_predictor()
            if predictor is None:
                flash("Model not trained. Run the training pipeline first.", "danger")
                return render_template("upload.html", batch_results=[])
            try:
                df = pd.read_csv(f)
                skipped = 0
                for _, row in df.iterrows():
                    try:
                        res = predictor.predict(row.to_dict())
                        results.append({
                            "student": str(row.get("student_id", f"Row {_ + 1}")),
                            **res,
                        })
                    except Exception:
                        skipped += 1
                msg = f"Processed {len(results)} students."
                if skipped:
                    msg += f" ({skipped} rows skipped — missing columns)"
                flash(msg, "success" if results else "warning")
                # store in session for export
                session["batch_results"] = results
            except Exception as e:
                flash(f"CSV Error: {e}", "danger")
    return render_template("upload.html", batch_results=results)


@app.route("/upload/export")
@login_required
def export_batch_csv():
    results = session.get("batch_results", [])
    if not results:
        flash("No batch results to export.", "warning")
        return redirect(url_for("upload_csv"))
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(["student_id","predicted_score","grade","risk_level","confidence"])
    for r in results:
        cw.writerow([r["student"], r["predicted_score"], r["grade"], r["risk_level"], r["confidence"]])
    out = io.BytesIO()
    out.write(si.getvalue().encode())
    out.seek(0)
    return send_file(out, mimetype="text/csv",
                     download_name="batch_predictions.csv", as_attachment=True)


@app.route("/upload/template")
@login_required
def download_template():
    """Download a blank CSV template with the correct column headers."""
    cols = ["student_id","gender","race_ethnicity","parental_education",
            "lunch","test_prep_course","extracurricular","internet_access",
            "study_hours_per_week","attendance_rate",
            "math_score","reading_score","writing_score","science_score"]
    sample = ["STU001","female","Group C","bachelor's degree",
              "standard","completed","yes","yes","6","90","85","88","82","84"]
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(cols)
    cw.writerow(sample)
    out = io.BytesIO()
    out.write(si.getvalue().encode())
    out.seek(0)
    return send_file(out, mimetype="text/csv",
                     download_name="student_template.csv", as_attachment=True)


def _plotly_response(fig):
    import plotly.utils

    return app.response_class(
        json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
        mimetype="application/json",
    )


def _plotly_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(color="#cdd6f4", size=15)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ba3c7"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=50, b=40),
    )


@app.route("/api/charts/score-distribution")
@login_required
def api_score_dist():
    df = load_dataset()
    if df.empty:
        return jsonify({"data": [], "layout": {}})
    import plotly.graph_objects as go

    colors = {
        "math_score": "#4a90e2",
        "reading_score": "#22c55e",
        "writing_score": "#f59e0b",
        "science_score": "#38bdf8",
        TARGET: "#ef4444",
    }
    traces = []
    for col, color in colors.items():
        if col in df.columns:
            label = FEATURE_LABELS.get(col, col.replace("_score", "").replace("_", " ").title())
            traces.append(go.Histogram(x=df[col], name=label, marker_color=color, opacity=0.72, nbinsx=20))
    fig = go.Figure(data=traces)
    fig.update_layout(**_plotly_layout("Current Scores vs Final Score"))
    return _plotly_response(fig)


@app.route("/api/charts/risk-pie")
@login_required
def api_risk_pie():
    df = load_dataset()
    col = score_column(df)
    if df.empty or not col:
        return jsonify({})
    import plotly.graph_objects as go

    risk = df["risk_level"] if "risk_level" in df.columns else df[col].apply(get_risk_level)
    counts = risk.value_counts()
    fig = go.Figure(
        go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.5,
            marker_colors=["#22c55e", "#f59e0b", "#ef4444"],
        )
    )
    fig.update_layout(**_plotly_layout("Risk Level Distribution"))
    return _plotly_response(fig)


@app.route("/api/charts/model-comparison")
@login_required
def api_model_comparison():
    metrics = load_metrics()
    if not metrics or "all_models" not in metrics:
        return jsonify({})
    import plotly.graph_objects as go

    names = list(metrics["all_models"].keys())
    r2s = [metrics["all_models"][n]["test_r2"] for n in names]
    maes = [metrics["all_models"][n]["mae"] for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="R2 Score", x=names, y=r2s, marker_color="#7c6af7"))
    fig.add_trace(go.Bar(name="MAE", x=names, y=maes, marker_color="#f38ba8"))
    fig.update_layout(barmode="group", **_plotly_layout("Model Performance Comparison"))
    return _plotly_response(fig)


@app.route("/api/charts/feature-importance")
@login_required
def api_feature_importance():
    metrics = load_metrics()
    items = metrics.get("feature_importance", [])[:10]
    if not items:
        return jsonify({})
    import plotly.graph_objects as go

    labels = [item["label"] for item in reversed(items)]
    values = [item["importance"] for item in reversed(items)]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color="#38bdf8"))
    fig.update_layout(**_plotly_layout("Top Prediction Drivers"))
    return _plotly_response(fig)


@app.route("/api/charts/risk-segments")
@login_required
def api_risk_segments():
    df = load_dataset()
    col = score_column(df)
    if df.empty or not col:
        return jsonify({})
    import plotly.graph_objects as go

    work = df.copy()
    if "risk_level" not in work.columns:
        work["risk_level"] = work[col].apply(get_risk_level)
    grouped = work.groupby("risk_level").agg(
        students=("risk_level", "size"),
        avg_score=(col, "mean"),
        avg_attendance=("attendance_rate", "mean"),
        avg_study=("study_hours_per_week", "mean"),
    )
    order = [name for name in ["Low", "Medium", "High"] if name in grouped.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Students", x=order, y=grouped.loc[order, "students"], marker_color="#7c6af7"))
    fig.add_trace(go.Bar(name="Avg Final Score", x=order, y=grouped.loc[order, "avg_score"], marker_color="#22c55e"))
    fig.add_trace(go.Bar(name="Avg Attendance", x=order, y=grouped.loc[order, "avg_attendance"], marker_color="#f59e0b"))
    fig.update_layout(barmode="group", **_plotly_layout("Risk Segment Snapshot"))
    return _plotly_response(fig)


@app.route("/api/charts/heatmap")
@login_required
def api_heatmap():
    df = load_dataset()
    if df.empty:
        return jsonify({})
    import plotly.graph_objects as go

    num_cols = [
        "math_score",
        "reading_score",
        "writing_score",
        "science_score",
        "study_hours_per_week",
        "attendance_rate",
        "current_average_score",
        TARGET,
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr().round(2)
    fig = go.Figure(
        go.Heatmap(
            z=corr.values.tolist(),
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=corr.values.round(2).tolist(),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(**_plotly_layout("Correlation Heatmap"))
    return _plotly_response(fig)


@app.route("/api/charts/gender-scores")
@login_required
def api_gender_scores():
    df = load_dataset()
    col = score_column(df)
    if df.empty or not col:
        return jsonify({})
    import plotly.graph_objects as go

    fig = go.Figure()
    for gender, color in [("male", "#4a90e2"), ("female", "#f38ba8")]:
        sub = df[df["gender"] == gender][col] if "gender" in df.columns else pd.Series(dtype=float)
        fig.add_trace(go.Box(y=sub, name=gender.title(), marker_color=color))
    fig.update_layout(**_plotly_layout("Final Score by Gender"))
    return _plotly_response(fig)


@app.route("/api/charts/attendance-scatter")
@login_required
def api_attendance():
    df = load_dataset()
    col = score_column(df)
    if df.empty or not col:
        return jsonify({})
    import plotly.graph_objects as go

    colors_map = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}
    if "risk_level" not in df.columns:
        df = df.copy()
        df["risk_level"] = df[col].apply(get_risk_level)
    fig = go.Figure()
    for level, color in colors_map.items():
        sub = df[df["risk_level"] == level]
        fig.add_trace(
            go.Scatter(
                x=sub.get("attendance_rate"),
                y=sub.get(col),
                mode="markers",
                name=f"Risk: {level}",
                marker=dict(color=color, size=6, opacity=0.7),
            )
        )
    fig.update_layout(**_plotly_layout("Attendance vs Final Score"))
    return _plotly_response(fig)


@app.route("/api/export/predictions")
@login_required
def export_predictions():
    rows = Prediction.query.filter_by(user_id=current_user.id).all()
    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(
        [
            "ID",
            "Student",
            "Score",
            "Interval Low",
            "Interval High",
            "Grade",
            "Risk",
            "Confidence",
            "Model Version",
            "Date",
        ]
    )
    for r in rows:
        writer.writerow(
            [
                r.id,
                r.student_name,
                r.predicted_score,
                r.prediction_low,
                r.prediction_high,
                r.grade,
                r.risk_level,
                r.confidence,
                r.model_version,
                r.created_at.strftime("%Y-%m-%d"),
            ]
        )
    output = io.BytesIO(si.getvalue().encode())
    output.seek(0)
    return send_file(output, mimetype="text/csv", download_name="predictions.csv", as_attachment=True)


@app.route("/api/export/all-predictions")
@login_required
@admin_required
def api_export_all():
    rows = Prediction.query.order_by(Prediction.created_at.desc()).all()
    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(
        [
            "ID",
            "User",
            "Student",
            "Score",
            "Interval Low",
            "Interval High",
            "Grade",
            "Risk",
            "Confidence",
            "Model Version",
            "Date",
        ]
    )
    for r in rows:
        username = r.user.username if r.user else "deleted"
        writer.writerow(
            [
                r.id,
                username,
                r.student_name,
                r.predicted_score,
                r.prediction_low,
                r.prediction_high,
                r.grade,
                r.risk_level,
                r.confidence,
                r.model_version,
                r.created_at.strftime("%Y-%m-%d"),
            ]
        )
    output = io.BytesIO(si.getvalue().encode())
    output.seek(0)
    return send_file(output, mimetype="text/csv", download_name="all_predictions.csv", as_attachment=True)


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    predictor = get_predictor()
    if predictor is None:
        return jsonify({"error": "Model not trained"}), 503
    try:
        return jsonify(predictor.predict(data))
    except InputValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.errors}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/report")
@login_required
def report():
    metrics = load_metrics()
    df = load_dataset()
    stats = dataset_stats(df)
    preds = (
        Prediction.query.filter_by(user_id=current_user.id)
        .order_by(Prediction.created_at.desc())
        .all()
    )
    return render_template("report.html", metrics=metrics, stats=stats, preds=preds, now=utc_now())


@app.route("/download-pdf")
@login_required
def download_pdf():
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=30, bottomMargin=30)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=20, textColor=colors.HexColor("#7c6af7"))
        elems = [
            Paragraph("AI Student Analytics Report", title_style),
            Spacer(1, 12),
            Paragraph(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | User: {current_user.username}",
                styles["Normal"],
            ),
            Spacer(1, 20),
        ]

        metrics = load_metrics()
        if metrics.get("all_models"):
            elems.append(Paragraph("Model Performance", styles["Heading2"]))
            rows = [["Model", "Test R2", "MAE", "RMSE"]]
            for name, m in metrics["all_models"].items():
                rows.append([name, str(m["test_r2"]), str(m["mae"]), str(m["rmse"])])
            table = Table(rows, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c6af7")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#1e1e2e"), colors.HexColor("#2a2a3d")]),
                        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#cdd6f4")),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#444466")),
                    ]
                )
            )
            elems += [table, Spacer(1, 20)]

        preds = (
            Prediction.query.filter_by(user_id=current_user.id)
            .order_by(Prediction.created_at.desc())
            .limit(20)
            .all()
        )
        if preds:
            elems.append(Paragraph("Recent Predictions", styles["Heading2"]))
            rows = [["Student", "Score", "Interval", "Grade", "Risk", "Confidence", "Date"]]
            for p in preds:
                interval = f"{p.prediction_low or ''}-{p.prediction_high or ''}"
                rows.append(
                    [
                        p.student_name,
                        str(p.predicted_score),
                        interval,
                        p.grade,
                        p.risk_level,
                        f"{p.confidence}%",
                        p.created_at.strftime("%Y-%m-%d"),
                    ]
                )
            table = Table(rows, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a90e2")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#1e1e2e"), colors.HexColor("#2a2a3d")]),
                        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#cdd6f4")),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#444466")),
                    ]
                )
            )
            elems.append(table)
        doc.build(elems)
        buf.seek(0)
        return send_file(buf, mimetype="application/pdf", download_name="analytics_report.pdf", as_attachment=True)
    except ImportError:
        flash("ReportLab not installed. Run: pip install reportlab", "warning")
        return redirect(url_for("report"))


# ── Error Handlers ──────────────────────────────────────────────────────────────
@app.errorhandler(403)
def forbidden(e):
    from flask_login import current_user
    if current_user.is_authenticated:
        return render_template("error.html",
            code=403, title="Access Denied",
            message="You don't have permission to view this page."), 403
    return render_template("login.html", error="403 — Access Denied"), 403

@app.errorhandler(404)
def not_found(e):
    from flask_login import current_user
    if current_user.is_authenticated:
        return render_template("error.html",
            code=404, title="Page Not Found",
            message="The page you're looking for doesn't exist."), 404
    return render_template("login.html", error="404 — Page Not Found"), 404

@app.errorhandler(500)
def server_error(e):
    from flask_login import current_user
    if current_user.is_authenticated:
        return render_template("error.html",
            code=500, title="Server Error",
            message=f"Something went wrong on our end. Error: {e}"), 500
    return render_template("login.html", error=f"500 — Server Error"), 500


def ensure_prediction_schema():
    """Add nullable columns to older SQLite databases."""
    inspector = inspect(db.engine)
    if "prediction" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("prediction")}
    additions = {
        "model_version": "VARCHAR(80)",
        "prediction_low": "FLOAT",
        "prediction_high": "FLOAT",
        "insights_json": "TEXT",
    }
    with db.engine.begin() as conn:
        for name, sql_type in additions.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE prediction ADD COLUMN {name} {sql_type}"))


def init_db():
    with app.app_context():
        db.create_all()
        ensure_prediction_schema()
        if CREATE_DEFAULT_ADMIN and User.query.count() == 0:
            admin_user = User(
                username=DEFAULT_ADMIN_USERNAME,
                email=DEFAULT_ADMIN_EMAIL,
                password=generate_password_hash(DEFAULT_ADMIN_PASSWORD),
                is_admin=True,
            )
            db.session.add(admin_user)
            db.session.commit()
            if os.environ.get("ADMIN_PASSWORD"):
                print(f"[DB] Default admin created: {DEFAULT_ADMIN_USERNAME} (password from ADMIN_PASSWORD)")
            else:
                print(f"[DB] Development admin created: {DEFAULT_ADMIN_USERNAME} / admin123")


if __name__ == "__main__":
    init_db()
    print("\n" + "=" * 52)
    print("  AI Student Analytics System")
    print("  URL:   http://127.0.0.1:5000")
    if admin_hint():
        print(f"  Login: {admin_hint()}")
    print("=" * 52 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
