"""API local para predecir el puntaje final de un estudiante."""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseModel


MODEL_PATH = Path(__file__).resolve().parent / "model" / "student-score-v1.joblib"
app = FastAPI(title="Student Final Score Prediction")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InputData(BaseModel):
    Student_ID: int = 1
    Semester_ID: int = 1
    Age: int = 25
    Gender: str = "Male"
    Region_Type: str = "Rural"
    Family_Size: int = 2
    Parent_Education: str = "Secondary"
    Family_Income_Level: str = "Middle"
    Internet_Quality: float = 1.8
    Study_Space_Quality: float = 7.7
    Previous_GPA: float = 2.71
    Number_of_Failed_Courses: int = 0
    Total_Credits_Earned: int = 17
    Weekly_Study_Hours: float = 37.2
    Attendance_Rate: float = 73.5
    Library_Visits_Per_Month: int = 8
    Extracurricular_Hours: float = 1.4
    Sleep_Hours: float = 6.8
    Social_Media_Usage_Hours: float = 2.5
    Stress_Level: float = 9.5
    Motivation_Score: float = 7.3
    Self_Efficacy_Score: float = 5.82
    Midterm_Mark: float = 54.3


class OutputData(BaseModel):
    predicted_final_exam_score: float


@app.on_event("startup")
def load_model() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No existe el modelo en {MODEL_PATH}. Ejecuta `python model/train.py` primero."
        )
    app.state.model = load(MODEL_PATH)

@app.post("/score", response_model=OutputData)
def score(data: InputData) -> OutputData:
    model_input = pd.DataFrame([data.model_dump()])
    prediction = app.state.model.predict(model_input)[0]
    return OutputData(predicted_final_exam_score=float(prediction))
