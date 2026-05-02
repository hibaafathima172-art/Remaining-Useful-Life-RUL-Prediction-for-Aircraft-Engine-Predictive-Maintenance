# ============================================================
# Aircraft Engine RUL Prediction Dashboard
# Streamlit App
#
# Features:
# 1. Simple user input
# 2. Auto-filled hidden model features
# 3. Healthy / Warning / Critical prediction
# 4. Prediction Summary
# 5. Key Drivers
# 6. Interpretation
# 7. Recommendation
#
# Required files in the same folder:
# - rul_model.pkl
# - scaler.pkl
# - features.pkl
# - default_profiles.pkl  optional but recommended
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="Aircraft Engine RUL Dashboard",
    layout="wide"
)


# ============================================================
# 2. Custom CSS
# ============================================================

def set_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }

        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: #ffffff !important;
        }

        .main-title {
            text-align: center;
            font-size: 42px;
            font-weight: 900;
            margin-bottom: 8px;
        }

        .subtitle {
            text-align: center;
            color: #d1d5db !important;
            font-size: 17px;
            margin-bottom: 24px;
        }

        .info-card {
            background-color: #1f2937;
            padding: 18px 22px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.28);
            margin-bottom: 24px;
            line-height: 1.7;
        }

        .panel-card {
            background-color: #111827;
            padding: 26px;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.25);
            box-shadow: 0px 8px 28px rgba(0,0,0,0.35);
            margin-bottom: 18px;
        }

        .result-card {
            background-color: #111827;
            padding: 30px;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.35);
            box-shadow: 0px 8px 28px rgba(0,0,0,0.45);
            margin-bottom: 22px;
        }

        .status-box {
            padding: 26px;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 22px;
        }

        .status-box h2 {
            margin: 0;
            font-size: 40px;
            font-weight: 900;
            letter-spacing: 1px;
        }

        .status-box p {
            margin: 8px 0 0 0;
            font-size: 18px;
            font-weight: 700;
        }

        .section-card {
            background-color: #1f2937;
            padding: 20px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.18);
            margin-bottom: 18px;
            line-height: 1.7;
        }

        .section-card h3 {
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 22px;
            font-weight: 900;
        }

        .section-card p {
            font-size: 16px;
            color: #e5e7eb !important;
        }

        .driver-row {
            background-color: #0f172a;
            padding: 12px 14px;
            border-radius: 12px;
            margin-bottom: 10px;
            border-left: 5px solid #ffffff;
        }

        [data-testid="stMetric"] {
            background-color: #1f2937;
            padding: 18px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.15);
        }

        [data-testid="stMetricValue"] {
            font-size: 36px !important;
            font-weight: 900 !important;
            color: #ffffff !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 16px !important;
            font-weight: 700 !important;
            color: #d1d5db !important;
        }

        .stButton > button {
            width: 100%;
            background-color: #2563eb;
            color: #ffffff;
            font-weight: 800;
            font-size: 17px;
            border-radius: 12px;
            padding: 0.8rem 1rem;
            border: none;
        }

        .stButton > button:hover {
            background-color: #1d4ed8;
            color: #ffffff;
        }

        div[data-baseweb="select"] > div {
            background-color: #111827 !important;
            color: #ffffff !important;
            border-color: rgba(255,255,255,0.3) !important;
        }

        input {
            background-color: #111827 !important;
            color: #ffffff !important;
        }

        .stNumberInput label,
        .stSelectbox label,
        .stFileUploader label {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #111827;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


set_custom_css()


# ============================================================
# 3. Load Model Files
# ============================================================

@st.cache_resource
def load_files():
    required_files = [
        "rul_model.pkl",
        "scaler.pkl",
        "features.pkl"
    ]

    missing_files = [
        file for file in required_files
        if not os.path.exists(file)
    ]

    if missing_files:
        st.error(
            "Missing file(s): "
            + ", ".join(missing_files)
            + ". Keep rul_model.pkl, scaler.pkl and features.pkl in the same folder as main.py."
        )
        st.stop()

    model = joblib.load("rul_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")

    if os.path.exists("default_profiles.pkl"):
        default_profiles = joblib.load("default_profiles.pkl")
    else:
        default_profiles = {}

    return model, scaler, features, default_profiles


model, scaler, features, default_profiles = load_files()


# ============================================================
# 4. Fallback Defaults
# Used if default_profiles.pkl is missing
# ============================================================

fallback_defaults = {
    "op1": 0.0000,
    "op2": 0.0000,
    "op3": 100.0000,

    "sensor1": 518.67,
    "sensor2": 642.68,
    "sensor3": 1590.52,
    "sensor4": 1408.93,
    "sensor5": 14.62,
    "sensor6": 21.61,
    "sensor7": 553.37,
    "sensor8": 2388.09,
    "sensor9": 9065.24,
    "sensor10": 1.30,
    "sensor11": 47.54,
    "sensor12": 521.41,
    "sensor13": 2388.09,
    "sensor14": 8143.75,
    "sensor15": 8.44,
    "sensor16": 0.03,
    "sensor17": 393.21,
    "sensor18": 2388.00,
    "sensor19": 100.00,
    "sensor20": 38.81,
    "sensor21": 23.29,
}


# ============================================================
# 5. Feature Labels
# ============================================================

feature_labels = {
    "op1": "Operational Setting 1",
    "op2": "Operational Setting 2",
    "op3": "Operational Setting 3",

    "sensor1": "Fan Inlet Temperature",
    "sensor2": "Compressor Temperature",
    "sensor3": "Compressor Pressure",
    "sensor4": "Turbine Temperature",
    "sensor5": "Fan Inlet Pressure",
    "sensor6": "Bypass Duct Pressure",
    "sensor7": "Fan Speed",
    "sensor8": "Physical Fan Speed",
    "sensor9": "Physical Core Speed",
    "sensor10": "Engine Pressure Ratio",
    "sensor11": "Exhaust Gas Temperature",
    "sensor12": "HPC Outlet Static Pressure",
    "sensor13": "Corrected Fan Speed",
    "sensor14": "Corrected Core Speed",
    "sensor15": "Bypass Ratio",
    "sensor16": "Burner Fuel-Air Ratio",
    "sensor17": "Bleed Enthalpy",
    "sensor18": "Required Fan Speed",
    "sensor19": "Required Fan Conversion Speed",
    "sensor20": "HPT Cool Air Flow",
    "sensor21": "LPT Cool Air Flow",
}


# Only these will be shown to the user by default
simple_user_inputs = {
    "sensor2": "Compressor Temperature",
    "sensor3": "Compressor Pressure",
    "sensor4": "Turbine Temperature",
    "sensor7": "Fan Speed",
    "sensor11": "Exhaust Gas Temperature",
    "sensor12": "HPC Outlet Static Pressure",
    "sensor15": "Bypass Ratio",
}


# These are used for the key drivers section
driver_features = [
    "sensor2",
    "sensor3",
    "sensor4",
    "sensor7",
    "sensor9",
    "sensor11",
    "sensor12",
    "sensor14",
    "sensor15",
    "sensor17",
    "sensor20",
    "sensor21",
]


# ============================================================
# 6. Helper Functions
# ============================================================

def get_status_details(rul):
    if rul > 80:
        return {
            "status": "HEALTHY",
            "risk": "LOW",
            "color": "#22c55e",
            "summary": "The engine is operating in a healthy range.",
            "interpretation": (
                "The predicted Remaining Useful Life is high. "
                "The current sensor pattern does not indicate severe degradation."
            ),
            "recommendation": (
                "Continue normal operation and routine monitoring. "
                "No immediate maintenance action is required."
            )
        }

    elif rul > 30:
        return {
            "status": "WARNING",
            "risk": "MEDIUM",
            "color": "#f59e0b",
            "summary": "The engine shows signs of degradation.",
            "interpretation": (
                "The predicted Remaining Useful Life is in the warning range. "
                "This means the engine is still usable, but sensor values suggest that degradation may be progressing."
            ),
            "recommendation": (
                "Schedule inspection, increase monitoring frequency, and plan preventive maintenance. "
                "Avoid delaying maintenance until the engine reaches the critical range."
            )
        }

    else:
        return {
            "status": "CRITICAL",
            "risk": "HIGH",
            "color": "#ef4444",
            "summary": "The engine is in a critical condition.",
            "interpretation": (
                "The predicted Remaining Useful Life is very low. "
                "The model indicates that the engine may be close to failure based on the current sensor pattern."
            ),
            "recommendation": (
                "Immediate maintenance action is recommended. "
                "Avoid continued operation unless cleared by technical inspection."
            )
        }


def get_active_profile(profile_name):
    if default_profiles and profile_name in default_profiles:
        return default_profiles[profile_name]

    return fallback_defaults


def build_complete_input_values(active_profile):
    """
    Creates a complete feature dictionary.
    Even though the user enters only a few values,
    the model still receives all trained features.
    """
    input_values = {}

    for feature in features:
        input_values[feature] = float(
            active_profile.get(
                feature,
                fallback_defaults.get(feature, 0.0)
            )
        )

    return input_values


def build_input_dataframe(input_values):
    """
    Builds dataframe in the exact same column order used during training.
    """
    row = {}

    for feature in features:
        row[feature] = float(
            input_values.get(
                feature,
                fallback_defaults.get(feature, 0.0)
            )
        )

    return pd.DataFrame([row], columns=features)


def predict_rul(input_values):
    input_df = build_input_dataframe(input_values)
    input_scaled = scaler.transform(input_df)

    raw_prediction = float(model.predict(input_scaled)[0])

    clipped_rul = int(
        np.clip(
            round(raw_prediction),
            0,
            130
        )
    )

    return raw_prediction, clipped_rul


def get_prediction_summary(rul, health_percentage, failure_risk, status_details):
    return (
        f"The model predicts a Remaining Useful Life of approximately "
        f"{rul} cycles. This places the engine in the "
        f"{status_details['status']} category with a "
        f"{status_details['risk']} failure risk. "
        f"The estimated health score is {health_percentage}%, "
        f"while the failure risk indicator is {failure_risk}%."
    )


def get_key_drivers(input_values, active_profile):
    """
    This is a practical dashboard-level driver explanation.
    It compares current input values against the selected/default profile.
    It does not claim true model explainability like SHAP.
    """
    rows = []

    for feature in driver_features:
        if feature in features:
            current_value = float(input_values.get(feature, 0.0))
            baseline_value = float(
                active_profile.get(
                    feature,
                    fallback_defaults.get(feature, 0.0)
                )
            )

            difference = current_value - baseline_value

            if abs(baseline_value) > 0:
                percent_change = (difference / abs(baseline_value)) * 100
            else:
                percent_change = 0

            rows.append({
                "Feature": feature,
                "Label": feature_labels.get(feature, feature),
                "Current Value": current_value,
                "Baseline Value": baseline_value,
                "Change": difference,
                "Percent Change": percent_change,
                "Abs Percent Change": abs(percent_change)
            })

    drivers_df = pd.DataFrame(rows)

    if drivers_df.empty:
        return drivers_df

    drivers_df = drivers_df.sort_values(
        by="Abs Percent Change",
        ascending=False
    ).head(5)

    return drivers_df


def get_driver_text(row):
    label = row["Label"]
    change = row["Change"]
    percent_change = row["Percent Change"]

    if abs(percent_change) < 0.01:
        direction = "remained close to the selected profile value"
    elif change > 0:
        direction = "is higher than the selected profile value"
    else:
        direction = "is lower than the selected profile value"

    return (
        f"{label} {direction}. "
        f"Change: {change:.2f} ({percent_change:.2f}%)."
    )


def get_interpretation_text(rul, status_details):
    if rul > 80:
        return (
            "The prediction suggests that the engine has a comfortable operating margin. "
            "The current sensor pattern is closer to a healthy operating condition, so the model does not detect urgent failure risk."
        )

    elif rul > 30:
        return (
            "The prediction suggests that the engine is entering a degradation zone. "
            "This does not mean immediate failure, but it indicates that maintenance planning should begin before the condition worsens."
        )

    else:
        return (
            "The prediction suggests that the engine has very limited useful life remaining. "
            "The sensor pattern is closer to end-of-life behaviour, so the engine should be treated as high priority for maintenance."
        )


def uploaded_csv_to_input_values(uploaded_file):
    uploaded_df = pd.read_csv(uploaded_file)

    missing_columns = [
        column for column in features
        if column not in uploaded_df.columns
    ]

    if missing_columns:
        st.error(
            "Uploaded CSV is missing these required columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    first_row = uploaded_df[features].iloc[0]
    return first_row.to_dict(), uploaded_df


# ============================================================
# 7. Header
# ============================================================

st.markdown(
    """
    <div class="main-title">
        Aircraft Engine RUL Prediction Dashboard
    </div>
    <div class="subtitle">
        Predict Remaining Useful Life using key engine sensor values
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
        <b>Health State Rules:</b><br>
        <b style="color:#22c55e !important;">HEALTHY:</b> RUL &gt; 80 &nbsp;&nbsp; | &nbsp;&nbsp;
        <b style="color:#f59e0b !important;">WARNING:</b> RUL 31–80 &nbsp;&nbsp; | &nbsp;&nbsp;
        <b style="color:#ef4444 !important;">CRITICAL:</b> RUL ≤ 30
        <br>
        <small>
        Users only need to enter key sensor values. The remaining trained model features are auto-filled in the background.
        </small>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 8. Layout
# ============================================================

left_col, right_col = st.columns([1, 2], gap="large")


# ============================================================
# 9. Input Panel
# ============================================================

with left_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)

    st.subheader("Input Method")

    input_method = st.radio(
        "Choose how you want to provide engine data",
        ["Simple Manual Input", "Upload CSV"],
        index=0
    )

    input_values = None
    active_profile = fallback_defaults
    uploaded_df = None

    if input_method == "Simple Manual Input":

        if default_profiles:
            profile_options = list(default_profiles.keys()) + ["Manual Custom"]
        else:
            profile_options = ["Manual Custom"]

        selected_profile = st.selectbox(
            "Select Engine Profile",
            profile_options,
            help="The selected profile auto-fills the hidden model values."
        )

        if selected_profile != "Manual Custom":
            active_profile = get_active_profile(selected_profile)
        else:
            active_profile = fallback_defaults

        input_values = build_complete_input_values(active_profile)

        st.markdown("### Key Inputs")

        for feature, label in simple_user_inputs.items():
            if feature in features:
                input_values[feature] = st.number_input(
                    label,
                    value=float(input_values[feature]),
                    format="%.4f",
                    key=f"simple_{selected_profile}_{feature}"
                )

        with st.expander("Advanced Inputs / Full Model Features", expanded=False):
            st.caption(
                "These values are auto-filled. Change them only if full manual control is needed."
            )

            for feature in features:
                if feature not in simple_user_inputs:
                    input_values[feature] = st.number_input(
                        feature_labels.get(feature, feature),
                        value=float(input_values[feature]),
                        format="%.4f",
                        key=f"advanced_{selected_profile}_{feature}"
                    )

    else:
        uploaded_file = st.file_uploader(
            "Upload engine sensor CSV",
            type=["csv"],
            help="CSV should contain the same columns as features.pkl."
        )

        if uploaded_file is not None:
            input_values, uploaded_df = uploaded_csv_to_input_values(uploaded_file)
            active_profile = input_values
            st.success("CSV uploaded successfully. Using the first row for prediction.")

            with st.expander("Preview Uploaded Data", expanded=False):
                st.dataframe(uploaded_df.head())

        else:
            st.info("Upload a CSV file to continue.")

    predict_button = st.button("Predict RUL")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 10. Output Panel
# ============================================================

with right_col:

    if not predict_button:
        st.info("Enter key values or upload CSV, then click Predict RUL.")

    elif input_values is None:
        st.warning("Please provide input values before prediction.")

    else:
        try:
            raw_prediction, rul = predict_rul(input_values)

            health_percentage = int(
                np.clip(
                    round((rul / 130) * 100),
                    0,
                    100
                )
            )

            failure_risk = 100 - health_percentage

            status_details = get_status_details(rul)

            status = status_details["status"]
            risk = status_details["risk"]
            color = status_details["color"]

            prediction_summary = get_prediction_summary(
                rul,
                health_percentage,
                failure_risk,
                status_details
            )

            drivers_df = get_key_drivers(input_values, active_profile)

            interpretation_text = get_interpretation_text(
                rul,
                status_details
            )

            # ====================================================
            # Main Result Card
            # ====================================================

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="status-box" style="background-color:{color};">
                    <h2>{status}</h2>
                    <p>Risk Level: {risk}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            metric_1, metric_2, metric_3 = st.columns(3)

            metric_1.metric(
                label="Predicted RUL",
                value=f"{rul} cycles"
            )

            metric_2.metric(
                label="Health",
                value=f"{health_percentage}%"
            )

            metric_3.metric(
                label="Failure Risk",
                value=f"{failure_risk}%"
            )

            st.caption(
                f"Raw model prediction before clipping: {raw_prediction:.2f}"
            )

            st.markdown("</div>", unsafe_allow_html=True)

            # ====================================================
            # Prediction Summary
            # ====================================================

            st.markdown(
                f"""
                <div class="section-card">
                    <h3>Prediction Summary</h3>
                    <p>{prediction_summary}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ====================================================
            # Key Drivers
            # ====================================================

            st.markdown(
                """
                <div class="section-card">
                    <h3>Key Drivers</h3>
                    <p>
                        These are the sensor values that changed the most compared with the selected profile or baseline.
                    </p>
                """,
                unsafe_allow_html=True
            )

            if drivers_df.empty:
                st.write("No key driver data available.")
            else:
                for _, row in drivers_df.iterrows():
                    driver_text = get_driver_text(row)

                    st.markdown(
                        f"""
                        <div class="driver-row" style="border-left-color:{color};">
                            <b>{row["Label"]}</b><br>
                            {driver_text}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown("</div>", unsafe_allow_html=True)

            # ====================================================
            # Interpretation
            # ====================================================

            st.markdown(
                f"""
                <div class="section-card">
                    <h3>Interpretation</h3>
                    <p>{interpretation_text}</p>
                    <p>{status_details["interpretation"]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ====================================================
            # Recommendation
            # ====================================================

            st.markdown(
                f"""
                <div class="section-card" style="border-left: 6px solid {color};">
                    <h3>Recommendation</h3>
                    <p>{status_details["recommendation"]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ====================================================
            # Charts
            # ====================================================

            chart_col_1, chart_col_2 = st.columns([1, 1], gap="large")

            with chart_col_1:
                gauge_fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=rul,
                        title={"text": "Remaining Useful Life"},
                        gauge={
                            "axis": {
                                "range": [0, 130],
                                "tickcolor": "white"
                            },
                            "bar": {"color": color},
                            "bgcolor": "#111827",
                            "borderwidth": 1,
                            "bordercolor": "white",
                            "steps": [
                                {
                                    "range": [0, 30],
                                    "color": "rgba(239,68,68,0.45)"
                                },
                                {
                                    "range": [31, 80],
                                    "color": "rgba(245,158,11,0.45)"
                                },
                                {
                                    "range": [81, 130],
                                    "color": "rgba(34,197,94,0.45)"
                                }
                            ],
                        }
                    )
                )

                gauge_fig.update_layout(
                    paper_bgcolor="#111827",
                    font={"color": "white"},
                    height=330,
                    margin=dict(l=20, r=20, t=50, b=20)
                )

                st.plotly_chart(
                    gauge_fig,
                    use_container_width=True
                )

            with chart_col_2:
                visible_sensor_features = [
                    feature for feature in simple_user_inputs.keys()
                    if feature in features
                ]

                sensor_labels = [
                    feature_labels.get(feature, feature)
                    for feature in visible_sensor_features
                ]

                sensor_values = [
                    input_values.get(feature, 0.0)
                    for feature in visible_sensor_features
                ]

                bar_fig = go.Figure(
                    go.Bar(
                        x=sensor_values,
                        y=sensor_labels,
                        orientation="h",
                        marker_color=color,
                        text=[
                            f"{value:.2f}"
                            for value in sensor_values
                        ],
                        textposition="auto"
                    )
                )

                bar_fig.update_layout(
                    title="Key Input Sensor Values",
                    paper_bgcolor="#111827",
                    plot_bgcolor="#111827",
                    font=dict(color="#ffffff"),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.12)"
                    ),
                    yaxis=dict(
                        autorange="reversed"
                    ),
                    height=330,
                    margin=dict(l=10, r=10, t=50, b=10)
                )

                st.plotly_chart(
                    bar_fig,
                    use_container_width=True
                )

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)
            st.warning(
                "Check that rul_model.pkl, scaler.pkl, features.pkl and default_profiles.pkl "
                "were generated from the same training code."
            )