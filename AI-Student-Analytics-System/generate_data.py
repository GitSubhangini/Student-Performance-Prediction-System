"""
Generate a synthetic student dataset for the advanced analytics app.

The current subject scores are input signals. The target is final_score, which
is correlated with those signals but not a direct average of them.
"""
from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd

from src.utils.schema import CATEGORICAL_OPTIONS


DEFAULT_ROWS = 1000
DEFAULT_OUTPUT = os.path.join("data", "student_data.csv")

EDU_SCORE = {
    "some high school": -2,
    "high school": 0,
    "some college": 2,
    "associate's degree": 3,
    "bachelor's degree": 5,
    "master's degree": 7,
}

RACE_CONTEXT = {
    "Group A": -1.5,
    "Group B": -0.5,
    "Group C": 0.0,
    "Group D": 1.0,
    "Group E": 1.5,
}


def classify_risk(score: float) -> str:
    if score < 50:
        return "High"
    if score < 65:
        return "Medium"
    return "Low"


def generate_student_data(
    n: int = DEFAULT_ROWS,
    output_path: str = DEFAULT_OUTPUT,
    seed: int = 42,
) -> pd.DataFrame:
    """Create and save a reproducible synthetic dataset."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    rows = []
    for i in range(1, n + 1):
        gender = random.choice(CATEGORICAL_OPTIONS["gender"])
        race = random.choice(CATEGORICAL_OPTIONS["race_ethnicity"])
        edu = random.choice(CATEGORICAL_OPTIONS["parental_education"])
        lunch = random.choice(CATEGORICAL_OPTIONS["lunch"])
        prep = random.choice(CATEGORICAL_OPTIONS["test_prep_course"])
        extra = random.choice(CATEGORICAL_OPTIONS["extracurricular"])
        internet = random.choice(CATEGORICAL_OPTIONS["internet_access"])

        study_h = round(float(np.clip(rng.normal(5.5, 2.8), 0, 40)), 1)
        attendance = round(float(np.clip(rng.normal(84, 11), 35, 100)), 1)

        base = 58
        base += EDU_SCORE[edu]
        base += RACE_CONTEXT[race]
        base += 5 if lunch == "standard" else -3
        base += 4 if prep == "completed" else 0
        base += study_h * 1.1
        base += (attendance - 80) * 0.18
        base += 2 if extra == "yes" else -1
        base += 2 if internet == "yes" else -3
        base += rng.normal(0, 7)

        math = round(float(np.clip(base + rng.normal(0, 7), 0, 100)), 0)
        reading = round(float(np.clip(base + rng.normal(3, 6), 0, 100)), 0)
        writing = round(float(np.clip(base + rng.normal(2, 6), 0, 100)), 0)
        science = round(float(np.clip(base + rng.normal(1, 7), 0, 100)), 0)
        current_average = round((math + reading + writing + science) / 4, 2)

        final = current_average * 0.62
        final += study_h * 1.35
        final += (attendance - 75) * 0.22
        final += 4 if prep == "completed" else -2
        final += 2.5 if extra == "yes" else -1
        final += 2 if internet == "yes" else -3.5
        final += 2 if lunch == "standard" else -2
        final += EDU_SCORE[edu] * 0.35
        final += rng.normal(0, 5.5)
        final_score = round(float(np.clip(final, 0, 100)), 2)

        rows.append(
            {
                "student_id": f"STU{i:04d}",
                "gender": gender,
                "race_ethnicity": race,
                "parental_education": edu,
                "lunch": lunch,
                "test_prep_course": prep,
                "study_hours_per_week": study_h,
                "attendance_rate": attendance,
                "extracurricular": extra,
                "internet_access": internet,
                "math_score": int(math),
                "reading_score": int(reading),
                "writing_score": int(writing),
                "science_score": int(science),
                "average_score": current_average,
                "current_average_score": current_average,
                "final_score": final_score,
                "risk_level": classify_risk(final_score),
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    df = generate_student_data()
    print(f"Dataset saved: {DEFAULT_OUTPUT} ({len(df)} rows)")
    print(df[["current_average_score", "final_score", "study_hours_per_week", "attendance_rate"]].describe().round(2))


if __name__ == "__main__":
    main()
