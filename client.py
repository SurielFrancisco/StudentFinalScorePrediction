"""Ejemplo de petición contra la API local."""

import requests


body = {
    "Student_ID": 1,
    "Semester_ID": 1,
    "Age": 25,
    "Gender": "Male",
    "Region_Type": "Rural",
    "Family_Size": 2,
    "Parent_Education": "Secondary",
    "Family_Income_Level": "Middle",
    "Internet_Quality": 1.8,
    "Study_Space_Quality": 7.7,
    "Previous_GPA": 2.71,
    "Number_of_Failed_Courses": 0,
    "Total_Credits_Earned": 17,
    "Weekly_Study_Hours": 37.2,
    "Attendance_Rate": 73.5,
    "Library_Visits_Per_Month": 8,
    "Extracurricular_Hours": 1.4,
    "Sleep_Hours": 6.8,
    "Social_Media_Usage_Hours": 2.5,
    "Stress_Level": 9.5,
    "Motivation_Score": 7.3,
    "Self_Efficacy_Score": 5.82,
    "Midterm_Mark": 54.3,
}

response = requests.post("http://127.0.0.1:8000/score", json=body, timeout=10)
response.raise_for_status()
print(response.json())
