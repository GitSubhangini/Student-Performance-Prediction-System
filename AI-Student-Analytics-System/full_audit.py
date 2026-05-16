"""full_audit_v2.py — Tests ALL routes including new ones"""
import sys, os
sys.path.insert(0, ".")
from app import app, init_db
init_db()

errors = []
passed = []

with app.test_client() as c:
    c.post("/login", data={"username":"admin","password":"admin123"}, follow_redirects=True)
    passed.append("POST /login")

    routes = [
        "/", "/dashboard", "/predict", "/report", "/admin", "/upload",
        "/api/charts/score-distribution", "/api/charts/risk-pie",
        "/api/charts/heatmap", "/api/charts/model-comparison",
        "/api/charts/gender-scores", "/api/charts/attendance-scatter",
        "/upload/template",
    ]
    for route in routes:
        r = c.get(route, follow_redirects=True)
        html = r.data.decode("utf-8", errors="replace")
        if r.status_code not in (200, 302):
            errors.append(f"GET {route} → {r.status_code}")
        elif any(x in html for x in ["TemplateSyntaxError","TemplateAssertionError","UndefinedError","BuildError"]):
            errors.append(f"GET {route} → TEMPLATE ERROR")
        else:
            passed.append(f"GET {route}")

    # POST predict
    r = c.post("/predict", data={
        "student_name":"Test", "gender":"male", "race_ethnicity":"Group C",
        "parental_education":"bachelor's degree", "lunch":"standard",
        "test_prep_course":"completed", "extracurricular":"yes",
        "internet_access":"yes", "study_hours":"6", "attendance":"90",
        "math_score":"80", "reading_score":"82", "writing_score":"78", "science_score":"81",
    }, follow_redirects=True)
    html = r.data.decode("utf-8", errors="replace")
    if r.status_code != 200 or "Prediction error" in html:
        errors.append("POST /predict → FAILED")
    else:
        passed.append("POST /predict")

    # Test 404
    r = c.get("/nonexistent-page-xyz")
    if r.status_code != 404:
        errors.append(f"404 handler returned {r.status_code}")
    else:
        passed.append("404 error handler")

print(f"\n{'='*55}")
print(f"  AUDIT v2:  {len(passed)} passed  |  {len(errors)} errors")
print(f"{'='*55}")
for p in passed: print(f"  PASS  {p}")
for e in errors: print(f"  FAIL  {e}")
print()
