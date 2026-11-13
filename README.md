# Student Social Media & Mental Health Prediction System

An end-to-end Machine Learning web application that predicts a student's **Mental Health Score** based on their social media usage habits and lifestyle factors.

---

## 📂 Project Structure

```
student-mental-health-project/
├── data/                        # Raw dataset (CSV)
│   └── student_social_media.csv
├── notebooks/                   # Jupyter Notebook for ML workflow
│   └── mental_health_model.ipynb
├── backend/                     # FastAPI backend
│   ├── main.py
│   ├── schemas.py
│   └── requirements.txt
├── frontend/                    # HTML/CSS/JS frontend
│   ├── index.html
│   ├── style.css
│   └── script.js
├── models/                      # Saved ML pipeline
│   └── mental_health_model.pkl
├── requirements.txt             # Root-level dependencies
├── .gitignore
└── README.md
```

---

## 🧰 Tech Stack

| Layer        | Technology                              |
|--------------|-----------------------------------------|
| ML           | Python, Pandas, NumPy, Scikit-learn     |
| Visualization| Matplotlib, Seaborn                     |
| Backend      | FastAPI, Pydantic, Uvicorn              |
| Frontend     | HTML5, CSS3, Vanilla JavaScript         |
| Serialization| Joblib / Pickle                         |
| Deployment   | Render                                  |

---

## 🚀 Setup & Run Locally

### 1. Clone the repository
```bash
git clone <repo-url>
cd student-mental-health-project
```

### 2. Install backend dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Run the FastAPI backend
```bash
uvicorn main:app --reload
```

### 4. Open the frontend
Open `frontend/index.html` in your browser, or serve with a simple HTTP server.

---

## 📊 Dataset

- **Records:** 5000 students
- **Features:** 12 input features (age, gender, platform usage, sleep, stress, etc.)
- **Target:** `Mental_Health_Score` (continuous, regression)

---

## 🤖 Models Trained

- Linear Regression
- K-Nearest Neighbors Regressor
- Decision Tree Regressor
- Random Forest Regressor *(Best performer)*

---

## 📡 API Endpoint

| Method | Endpoint   | Description              |
|--------|------------|--------------------------|
| POST   | `/predict` | Predict mental health score |
| GET    | `/`        | Health check              |

---

## ☁️ Deployment (Render)

Configure the Render service with:
- **Build command:** `pip install -r backend/requirements.txt`
- **Start command:** `uvicorn backend.main:app --host 0.0.0.0 --port 10000`

