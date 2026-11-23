# ☄️ Asteroid Hazard Classification Using Machine Learning

An interactive machine-learning application for analysing and classifying near-Earth asteroid data.

## 📌 Project Overview

This project explores the use of machine learning to classify near-Earth asteroids into two categories:

- Potentially Hazardous
- Not Hazardous

It uses asteroid physical characteristics, orbital parameters and close-approach information to train and evaluate classification models.

The project also includes an interactive Streamlit dashboard for exploring the dataset and viewing model predictions.

**Note:** This is an independent educational project, not an official NASA application or an operational asteroid warning system.

## 🎯 Project Objectives

- Explore publicly available near-Earth asteroid data.
- Prepare and analyse asteroid-related features.
- Build machine-learning classification models.
- Compare model performance.
- Develop an interactive web application for predictions and data exploration.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Google Colab
- GitHub

## 🤖 Machine Learning Models

Two classification models are included:

### 1. Random Forest Classifier
Uses selected physical, orbital and close-approach features to classify asteroid records.

### 2. Decision Tree Classifier
Uses absolute magnitude and Minimum Orbit Intersection Distance (MOID) for classification.

These models use different feature sets. The Decision Tree features are closely related to conventional potentially hazardous asteroid classification criteria, so its results should not be interpreted as independent discovery of asteroid danger.

## 📊 Dataset Information

The dataset contains 200 asteroid records.

| Category | Records |
|---|---:|
| Total Asteroids | 200 |
| Potentially Hazardous | 48 |
| Not Hazardous | 152 |

The dataset includes information such as:

- Asteroid name and ID
- Absolute magnitude
- Estimated diameter
- Relative velocity
- Close-approach distance
- Orbital parameters
- Hazard classification label

## 🌐 Application Features

- Interactive project dashboard
- Asteroid classification using trained models
- Dataset exploration
- Model performance information
- Hazard classification distribution
- Downloadable dataset

## 📁 Project Structure

```text
asteroid-hazard-classification/
│
├── app.py
├── requirements.txt
├── asteroid_dataset.csv
├── random_forest_model.pkl
├── decision_tree_model.pkl
├── model_results.json
├── README.md
└── .gitignore
```

## ▶️ How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/asteroid-hazard-classification.git
```

### Step 2: Open the project folder

```bash
cd asteroid-hazard-classification
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Start the application

```bash
streamlit run app.py
```

## 🛰️ Data Source

Asteroid data is obtained from NASA/JPL's Near-Earth Object Web Service (NeoWs).

Website: https://api.nasa.gov/

## ⚠️ Disclaimer

This project is intended for educational and learning purposes only.

Machine-learning predictions should not be treated as actual impact probabilities or professional assessments of asteroid threats. The dataset is limited, and model performance does not establish real-world predictive reliability.

## Data Source & Acknowledgement

This project uses asteroid-related data obtained from NASA's Near Earth Object Web Service (NeoWs) API.

**Data Source:** https://api.nasa.gov/

This is an independent educational project developed for learning and experimentation. It is not officially affiliated with or endorsed by NASA.

## 👩‍💻 Author

Developed as an independent academic and machine-learning project.

**Project:** Asteroid Hazard Classification Using Machine Learning

**Domain:** Machine Learning | Data Science | Space Science
