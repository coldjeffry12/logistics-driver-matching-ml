"""Streamlit user interface for the logistics matching prototype."""

from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from src.config import CITY_COORDINATES, VEHICLE_CAPACITY_RANGES
    from src.recommender import DriverRecommender
    from src.runtime_setup import ensure_runtime_artifacts
    from src.utils import road_distance_km
except ModuleNotFoundError:
    from config import CITY_COORDINATES, VEHICLE_CAPACITY_RANGES
    from recommender import DriverRecommender
    from runtime_setup import ensure_runtime_artifacts
    from utils import road_distance_km

st.set_page_config(
    page_title="Logistics Driver Matching",
    page_icon="🚚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_recommender() -> DriverRecommender:
    ensure_runtime_artifacts()
    return DriverRecommender()


st.title("🚚 Logistics Recommendation & Driver Matching")
st.caption(
    "Portfolio prototype using synthetic data and a scikit-learn matching model."
)
st.info(
    "First cloud recommendation: allow about one minute if synthetic data and "
    "model artifacts need to be generated."
)

with st.sidebar:
    st.header("About")
    st.write(
        "The model combines compatibility, distance, availability, quality, "
        "risk, response time, and price features. Capacity and exact vehicle "
        "type are enforced before ranking, and offline drivers are excluded."
    )
    st.info("This is an educational prototype, not a production dispatch system.")

cities = list(CITY_COORDINATES)
vehicle_types = list(VEHICLE_CAPACITY_RANGES)

with st.form("shipment_form"):
    st.subheader("Shipment details")
    first, second, third = st.columns(3)
    with first:
        origin = st.selectbox("Origin city", cities, index=0)
        destination = st.selectbox("Destination city", cities, index=6)
        category = st.selectbox(
            "Shipment category",
            ["documents", "electronics", "food", "retail", "industrial", "furniture"],
            index=3,
        )
    with second:
        vehicle = st.selectbox("Required vehicle", vehicle_types, index=1)
        default_capacity = VEHICLE_CAPACITY_RANGES[vehicle][0]
        weight = st.number_input(
            "Shipment weight (kg)",
            min_value=1.0,
            value=float(max(20, default_capacity)),
            step=10.0,
        )
        urgency = st.selectbox(
            "Delivery urgency",
            ["standard", "express", "same_day"],
        )
    with third:
        route_distance = road_distance_km(origin, destination, CITY_COORDINATES)
        distance = st.number_input(
            "Route distance (km)",
            min_value=1.0,
            value=float(round(route_distance, 1)),
            step=5.0,
        )
        offered_price = st.number_input(
            "Offered price (MYR)",
            min_value=1.0,
            value=float(round(max(50, route_distance * 1.5), 2)),
            step=10.0,
        )
        pickup_hour = st.slider("Pickup hour", 0, 23, 10)
    submitted = st.form_submit_button(
        "Recommend drivers",
        type="primary",
        width="stretch",
    )

if submitted:
    if origin == destination:
        st.warning("Choose different origin and destination cities.")
        st.stop()
    shipment = {
        "origin_city": origin,
        "destination_city": destination,
        "distance_km": distance,
        "shipment_weight_kg": weight,
        "required_vehicle_type": vehicle,
        "delivery_urgency": urgency,
        "offered_price": offered_price,
        "pickup_hour": pickup_hour,
        "shipment_category": category,
    }
    try:
        with st.spinner(
            "Preparing the synthetic demo. The first cloud run may take about "
            "one minute while data and the model are generated."
        ):
            recommender = load_recommender()
        recommendations = recommender.recommend(shipment, top_k=5)
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.code("python src/data_generation.py\npython src/train_model.py")
        st.stop()

    if not recommendations:
        st.warning(
            "No non-offline driver currently meets the vehicle and capacity "
            "requirements."
        )
        st.stop()

    st.subheader("Top recommendations")
    st.caption(
        "Driver IDs such as DRV-0111 are synthetic identifiers created for "
        "this portfolio dataset."
    )
    result_frame = pd.DataFrame(recommendations)
    metrics = st.columns(min(5, len(result_frame)))
    for position, (_, recommendation) in enumerate(result_frame.iterrows()):
        with metrics[position]:
            st.metric(
                f"#{position + 1} {recommendation['driver_id']}",
                f"{recommendation['predicted_match_score']:.1%}",
                f"⭐ {recommendation['driver_rating']:.2f}",
            )

    chart_data = result_frame.set_index("driver_id")["predicted_match_score"]
    st.bar_chart(chart_data, color="#2E86AB")

    display_columns = [
        "driver_id",
        "predicted_match_score",
        "vehicle_type",
        "vehicle_capacity_kg",
        "driver_rating",
        "availability_status",
        "current_city",
        "pickup_distance_km",
        "estimated_driver_price",
    ]
    st.dataframe(
        result_frame[display_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "predicted_match_score": st.column_config.ProgressColumn(
                "Match score",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
            "estimated_driver_price": st.column_config.NumberColumn(
                "Est. driver price (MYR)",
                format="RM %.2f",
            ),
        },
    )

    st.subheader("Why these drivers?")
    for position, recommendation in enumerate(recommendations, start=1):
        st.markdown(
            f"**{position}. {recommendation['driver_id']}** — "
            f"{recommendation['reason']}"
        )
else:
    st.info("Enter a shipment and select **Recommend drivers** to score candidates.")
