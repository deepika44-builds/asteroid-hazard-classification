
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

st.set_page_config(
    page_title="Asteroid Classification",
    page_icon="☄️",
    layout="wide"
)

PROJECT_DIR = Path(__file__).parent

# Load project files
@st.cache_resource
def load_models():
    rf = joblib.load(PROJECT_DIR / "random_forest_model.pkl")
    dt = joblib.load(PROJECT_DIR / "decision_tree_model.pkl")
    return rf, dt

@st.cache_data
def load_data():
    df = pd.read_csv(PROJECT_DIR / "asteroid_dataset.csv")
    with open(PROJECT_DIR / "model_results.json", "r") as f:
        results = json.load(f)
    return df, results

rf_model, dt_model = load_models()
df, results = load_data()

# CSS styling
st.markdown("""
<style>
.stApp {
    background-color: #0b1020;
    color: #ffffff;
}
[data-testid="stSidebar"] {
    background-color: #151d32;
}
h1, h2, h3 {
    color: #8ab4ff;
}
div[data-testid="stMetric"] {
    background-color: #18243d;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("☄️ Asteroid Hazard Classification")
st.caption("Machine Learning Based Near-Earth Asteroid Analysis")

st.info(
    "Educational machine-learning project using publicly available "
    "near-Earth asteroid data. This application is not an official "
    "astronomical warning system."
)

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Asteroid Prediction",
        "Dataset Explorer",
        "Model Performance",
        "About Project"
    ]
)

# Dashboard
if page == "Dashboard":

    st.header("Project Dashboard")

    c1, c2, c3 = st.columns(3)

    total = len(df)

    target = "hazard_label" if "hazard_label" in df.columns else "is_hazardous"

    hazardous = int((df[target] == True).sum())

    c1.metric("Total Asteroids", total)
    c2.metric("Potentially Hazardous", hazardous)
    c3.metric("Other Asteroids", total - hazardous)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Hazard Classification Distribution")

    chart_data = (
        df[target]
        .astype(str)
        .value_counts()
        .rename(index={"True": "Potentially Hazardous", "False": "Not Hazardous"})
    )

    st.bar_chart(chart_data)

# Prediction
elif page == "Asteroid Prediction":

    st.header("Asteroid Classification")

    st.write(
        "Enter asteroid measurements below to obtain model classifications."
    )

    target = "hazard_label" if "hazard_label" in df.columns else "is_hazardous"

    rf_features = results.get(
        "model_a_features",
        results.get("model_a_features", [])
    )

    dt_features = results.get("model_b_features", [])

    # Use model feature names when available
    if not rf_features:
        rf_features = list(getattr(rf_model, "feature_names_in_", []))

    if not dt_features:
        dt_features = list(getattr(dt_model, "feature_names_in_", []))

    all_features = list(dict.fromkeys(rf_features + dt_features))

    if not all_features:
        st.error("Model feature information could not be loaded.")
        st.stop()

    with st.form("prediction_form"):

        st.subheader("Enter Asteroid Details")

        values = {}

        columns = st.columns(2)

        for i, feature in enumerate(all_features):

            if feature not in df.columns:
                continue

            default_value = pd.to_numeric(
                df[feature], errors="coerce"
            ).median()

            if pd.isna(default_value):
                default_value = 0.0

            with columns[i % 2]:
                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=float(default_value),
                    format="%.6f"
                )

        submitted = st.form_submit_button(
            "Classify Asteroid",
            use_container_width=True
        )

    if submitted:

        if rf_features:
            rf_input = pd.DataFrame(
                [[values[f] for f in rf_features]],
                columns=rf_features
            )

            rf_prediction = rf_model.predict(rf_input)[0]

            st.subheader("Random Forest Result")

            if bool(rf_prediction):
                st.warning("Potentially Hazardous")
            else:
                st.success("Not Hazardous")

            if hasattr(rf_model, "predict_proba"):
                probabilities = rf_model.predict_proba(rf_input)[0]
                classes = list(rf_model.classes_)

                if True in classes:
                    score = probabilities[classes.index(True)]
                    st.metric(
                        "Model score for hazardous class",
                        f"{score * 100:.2f}%"
                    )

        if dt_features:
            dt_input = pd.DataFrame(
                [[values[f] for f in dt_features]],
                columns=dt_features
            )

            dt_prediction = dt_model.predict(dt_input)[0]

            st.subheader("Decision Tree Result")

            if bool(dt_prediction):
                st.warning("Potentially Hazardous")
            else:
                st.success("Not Hazardous")

    st.caption(
        "Predictions are educational model outputs, not actual impact-risk "
        "probabilities or professional hazard assessments."
    )

# Dataset Explorer
elif page == "Dataset Explorer":

    st.header("Dataset Explorer")

    st.write("Explore the asteroid dataset.")

    st.metric("Total Records", len(df))

    search = st.text_input("Search asteroid by name")

    filtered_df = df.copy()

    if search:
        name_column = "name" if "name" in df.columns else None

        if name_column:
            filtered_df = filtered_df[
                filtered_df[name_column].astype(str).str.contains(
                    search, case=False, na=False
                )
            ]

    st.dataframe(filtered_df, use_container_width=True)

    st.download_button(
        "Download Dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="asteroid_dataset.csv",
        mime="text/csv"
    )

# Model Performance
elif page == "Model Performance":

    st.header("Model Performance")

    rf_results = results.get(
        "random_forest",
        results.get("model_a_random_forest", {})
    )

    dt_results = results.get(
        "decision_tree",
        results.get("model_b_decision_tree", {})
    )

    st.subheader("Random Forest")

    if isinstance(rf_results, dict):
        st.json(rf_results)
    else:
        st.write(rf_results)

    st.divider()

    st.subheader("Decision Tree")

    if isinstance(dt_results, dict):
        st.json(dt_results)
    else:
        st.write(dt_results)

    st.warning(
        "Model scores depend on the dataset and test split. "
        "They should not be interpreted as proof of real-world prediction accuracy."
    )

# About
elif page == "About Project":

    st.header("About the Project")

    st.write("""
    **Asteroid Hazard Classification Using Machine Learning**

    This project explores how machine-learning algorithms can classify
    near-Earth asteroid records into potentially hazardous and
    not-hazardous categories.

    ### Machine Learning Models
    - Random Forest Classifier
    - Decision Tree Classifier

    ### Technologies Used
    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    - NASA/JPL Near-Earth Object data

    ### Data Source
    NASA/JPL Center for Near-Earth Object Studies (CNEOS).

    This is an educational project and is not an official NASA application.
    """)
