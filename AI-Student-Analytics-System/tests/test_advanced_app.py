import csv
import io
import os
import tempfile
import unittest

os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(
    tempfile.gettempdir(), "studentai_test.sqlite"
).replace("\\", "/")
os.environ["SECRET_KEY"] = "test-secret"
os.environ["ADMIN_PASSWORD"] = "admin123"

from app import Prediction, app, db, init_db
from generate_data import generate_student_data
from src.pipeline.train_pipeline import run_training
from src.components.prediction_pipeline import PredictPipeline
from src.utils.helper import get_grade, get_risk_level
from src.utils.schema import InputValidationError, TARGET, validate_student_input


def sample_payload():
    return {
        "gender": "male",
        "race_ethnicity": "Group C",
        "parental_education": "bachelor's degree",
        "lunch": "standard",
        "test_prep_course": "completed",
        "extracurricular": "yes",
        "internet_access": "yes",
        "study_hours_per_week": 6,
        "attendance_rate": 88,
        "math_score": 72,
        "reading_score": 75,
        "writing_score": 73,
        "science_score": 74,
    }


def sample_form():
    data = sample_payload()
    data["study_hours"] = data.pop("study_hours_per_week")
    data["attendance"] = data.pop("attendance_rate")
    data["student_name"] = "Test Student"
    return data


class AdvancedAppTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = generate_student_data(n=1000, seed=123)
        cls.metrics = run_training()
        cls.generated_columns = set(df.columns)
        app.config.update(TESTING=True)
        with app.app_context():
            db.drop_all()
            init_db()
        if hasattr(app, "_predictor"):
            delattr(app, "_predictor")

    def setUp(self):
        with app.app_context():
            Prediction.query.delete()
            db.session.commit()
        self.client = app.test_client()

    def login(self):
        return self.client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            follow_redirects=True,
        )

    def test_training_smoke_outputs_final_score_metadata(self):
        self.assertIn(TARGET, self.generated_columns)
        self.assertEqual(self.metrics["target"], TARGET)
        self.assertIn("feature_importance", self.metrics)
        self.assertTrue(os.path.exists(os.path.join("artifacts", "model.pkl")))
        self.assertTrue(os.path.exists(os.path.join("artifacts", "preprocessor.pkl")))

    def test_helpers_and_validation(self):
        self.assertEqual(get_grade(91)[0], "A+")
        self.assertEqual(get_risk_level(49), "High")
        cleaned = validate_student_input(sample_payload())
        self.assertEqual(cleaned["race_ethnicity"], "Group C")
        bad = sample_payload()
        bad["attendance_rate"] = 150
        with self.assertRaises(InputValidationError):
            validate_student_input(bad)

    def test_prediction_result_shape(self):
        result = PredictPipeline().predict(sample_payload())
        for key in [
            "predicted_score",
            "grade",
            "risk_level",
            "confidence",
            "interval",
            "drivers",
            "recommendations",
            "model",
        ]:
            self.assertIn(key, result)
        self.assertLessEqual(result["interval"]["low"], result["predicted_score"])
        self.assertGreaterEqual(result["interval"]["high"], result["predicted_score"])
        self.assertTrue(result["drivers"])
        self.assertTrue(result["recommendations"])

    def test_login_dashboard_admin_and_exports(self):
        response = self.login()
        self.assertEqual(response.status_code, 200)
        for path in ["/dashboard", "/admin", "/report", "/api/export/predictions"]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)

    def test_form_and_api_prediction(self):
        self.login()
        response = self.client.post("/predict", data=sample_form(), follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction saved", response.data)

        response = self.client.post("/api/predict", json=sample_payload())
        self.assertEqual(response.status_code, 200)
        self.assertIn("drivers", response.get_json())

        bad = sample_payload()
        bad["math_score"] = -5
        response = self.client.post("/api/predict", json=bad)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Validation failed")

    def test_csv_upload_reports_valid_and_invalid_rows(self):
        self.login()
        fieldnames = list(sample_payload().keys())
        rows = [sample_payload(), {**sample_payload(), "attendance_rate": 180}]
        text = io.StringIO()
        writer = csv.DictWriter(text, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        data = {"file": (io.BytesIO(text.getvalue().encode("utf-8")), "students.csv")}

        response = self.client.post("/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Processed 1 rows with 1 validation issue", response.data)


if __name__ == "__main__":
    unittest.main()
