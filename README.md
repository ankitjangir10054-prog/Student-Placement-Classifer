# Student-Placement-Classifer
```markdown
# Student Placement Predictor (1-Million Row Dataset)

An end-to-end machine learning and deep learning repository containing analysis, feature engineering, and model deployment strategies for predicting university student placements. Utilizing a massive dataset of 1,000,000 students, this project implements Classical Machine Learning classifiers (XGBoost, Random Forest) alongside Optimized PyTorch Artificial Neural Networks (ANN).

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Description](#-dataset-description)
3. [Key Insights & Exploratory Data Analysis (EDA)](#-key-insights--exploratory-data-analysis-eda)
4. [The 67% Mathematical Performance Limit (Bayes Error)](#-the-67-mathematical-performance-limit-bayes-error)
5. [Feature Engineering Pipeline](#-feature-engineering-pipeline)
6. [Model Architectures & Performance](#-model-architectures--performance)
7. [Production Deployment Strategy](#-production-deployment-strategy)
8. [How to Run Local & Train](#-how-to-run-local--train)

---

## 📝 Project Overview
This project aims to predict whether a student will be placed in a campus recruitment drive based on their academic marks, technical skill levels, soft skills, participation in extracurriculars, and socioeconomic background. 

* **Scale:** 1,000,000 student profiles.
* **Objective:** Binary Classification (`placed`: `0` or `1`).
* **Frameworks Used:** PyTorch, XGBoost, Scikit-Learn, Pandas, Seaborn.

---

## 📊 Dataset Description
The dataset consists of 30 columns reflecting the multidimensional nature of student profiles:
* **Academics:** `cgpa`, `tenth_percentage`, `twelfth_percentage`, `backlogs`.
* **Technical Skills:** `python_skill`, `c++_skill`, `java_skill`, `ml_skill`, `web_dev_skill` (all rated on a 0-9 scale), `coding_contest_rating`.
* **Extracurriculars:** `internships`, `projects`, `github_projects`, `hackathons`, `certifications`.
* **Soft Skills:** `communication_skill`, `teamwork`, `leadership`, `problem_solving`, `time_management`.
* **Contextual & Socioeconomic:** `age`, `gender`, `family_income`, `college_tier`, `city_tier`, `branch`.

---

## 📈 Key Insights & Exploratory Data Analysis (EDA)
Through comprehensive visualization and exploration, we uncovered critical properties of the synthetic data generation engine:
* **Uniformly Sampled Categoricals:** Features like `gender` and engineering `branch` have absolutely uniform distributions and zero statistical correlation with placement rates (~50.0% placement rate across all subgroups).
* **Low Feature-Target Linear Relationships:** Feature correlation with the target is extremely weak (highest individual feature linear correlation was `python_skill`/`java_skill` at `0.14`).
* **Primary Predictors:** The technical skill sets, internships, and total projects represent the primary driving parameters behind higher placement likelihoods.

---

## 🎯 The 67% Mathematical Performance Limit (Bayes Error)
During experimental evaluations, both high-capacity tree classifiers (XGBoost) and deep multi-layer neural networks (ANN) consistently hit a performance ceiling of **~67.18% test accuracy**.

### Why?
This dataset contains a high level of **inherent noise and overlap (Bayes Error)**. Many students with identical technical skills, high CGPA, and positive metrics are labeled as unplaced, while students with identical profiles are labeled as placed. Because there are no distinct mathematical boundaries separating the two classes in the raw coordinate space, **67% represents the mathematical upper bound of predictability** on this specific dataset.

---

## 🛠 Feature Engineering Pipeline
To help the models map non-linear combinations of factors, we engineered highly effective domain features:

1. **Super Placement Score:** A weighted summation of scaled core drivers:
   $$\text{Super Placement Score} = \text{Standardized}(\text{Skills} + \text{Internships} + \text{Projects} + \text{Soft Skills} + \text{Teamwork} + \text{Problem Solving} + \text{Leadership})$$
   * This alone yielded a correlation of **0.3304** with the placement target.
2. **Multiplicative Interactions:** 
   * `cgpa_x_skills` (academic performance scaled by technical execution)
   * `internships_x_projects` (applied experience weight)
   * `cgpa_x_attendance` (presence-adjusted educational consistency)

---

## 🧠 Model Architectures & Performance

### 1. XGBoost (Full Raw Features)
* **Parameters:** `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`.
* **Test Accuracy:** **67.18%**
* **Precision / Recall / F1-Score:** Balanced at `0.67` across both classes.

### 2. Deep PyTorch ANN (With Scaled Interaction Terms)
An optimized feedforward architecture built to prevent overfitting on the synthetic noise:
```python
class OptimizedPlacementANN(nn.Module):
    def __init__(self, input_dim):
        super(OptimizedPlacementANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 1) # Raw Logits output
        )
```
* **Loss Function:** `BCEWithLogitsLoss()`
* **Optimizer:** `AdamW` (learning rate: `0.005`, weight decay: `1e-4`)
* **Regularization:** Batch Normalization, Dropout, early stopping validation recovery.
* **Test Accuracy:** **66.98%**

---

## 🚀 Production Deployment Strategy

We structure our deployment into a microservice pattern using **FastAPI** and containerization:

```
  [ Client / UI ] ───(JSON HTTP POST)───> [ FastAPI Gateway ] ───> [ PyTorch TorchScript Model ]
```

### Sample FastAPI Endpoint Script
```python
from fastapi import FastAPI
import torch
import numpy as np
from pydantic import BaseModel

# Define the API input structure
class StudentProfile(BaseModel):
    super_placement_score: float
    skills: float
    soft_skills: float
    total_projects: float
    internships: float
    teamwork: float
    problem_solving: float
    leadership: float
    cgpa_x_skills: float
    internships_x_projects: float
    skills_x_teamwork: float
    cgpa_x_attendance: float

app = FastAPI(title="Student Placement API", version="1.0")

# Load Model weights
model = OptimizedPlacementANN(input_dim=12)
model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cpu')))
model.eval()

@app.post("/predict")
def predict_placement(profile: StudentProfile):
    features = np.array([[v for k, v in profile.dict().items()]])
    # Scale features using pre-saved scaling vector
    features_scaled = (features - scaler_mean) / scaler_std
    tensor_features = torch.tensor(features_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        logits = model(tensor_features)
        probability = torch.sigmoid(logits).item()
        prediction = 1 if probability >= 0.5 else 0
        
    return {
        "placement_prediction": prediction,
        "placement_probability": round(probability, 4)
    }
```

---

## 💻 How to Run Local & Train

### 1. Setup Environment
```bash
git clone https://github.com/your-username/student-placement-classifier.git
cd student-placement-classifier
pip install -r requirements.txt
```

### 2. Download Dataset
Ensure you have your `kaggle.json` credentials configured:
```bash
kaggle datasets download -d sharmajicoder/student-placement-prediction
unzip student-placement-prediction.zip
```

### 3. Run Pipeline Notebook/Script
```bash
jupyter notebook pipeline_student_placement.ipynb
```
```
