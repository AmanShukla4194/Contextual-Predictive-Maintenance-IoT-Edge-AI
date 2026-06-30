import streamlit as st
import plotly.express as px
from config import SENSOR_COLS


def render(telemetry_df):
    st.header("Sensor Trends")

    if telemetry_df is None or telemetry_df.empty:
        st.warning("No telemetry data available.")
        return

    machines = sorted(telemetry_df["machineID"].unique())
    selected_machine = st.selectbox("Select Machine", machines)
    selected_sensor = st.selectbox("Select Sensor", SENSOR_COLS)

    machine_data = telemetry_df[telemetry_df["machineID"] == selected_machine].copy()
    machine_data = machine_data.sort_values("datetime")

    fig = px.line(
        machine_data,
        x="datetime",
        y=selected_sensor,
        title=f"Machine {selected_machine} — {selected_sensor} over time",
        labels={"datetime": "Timestamp", selected_sensor: selected_sensor.capitalize()},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(machine_data[SENSOR_COLS].describe().T.round(3))
