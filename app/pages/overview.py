import streamlit as st
import plotly.express as px


def render(telemetry_df, failures_df, machines_df):
    st.header("Fleet Overview")

    if telemetry_df is None or machines_df is None:
        st.warning("Data not loaded.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Machines", machines_df["machineID"].nunique())
    col2.metric("Total Records", f"{len(telemetry_df):,}")
    col3.metric("Total Failures", len(failures_df) if failures_df is not None else 0)

    st.subheader("Machine Age Distribution")
    fig_age = px.histogram(
        machines_df, x="age", nbins=20,
        title="Machine Age (years)",
        labels={"age": "Age (years)", "count": "Machines"}
    )
    st.plotly_chart(fig_age, use_container_width=True)

    if failures_df is not None and not failures_df.empty:
        st.subheader("Failures by Component")
        fail_counts = failures_df["failure"].value_counts().reset_index()
        fail_counts.columns = ["Component", "Count"]
        fig_fail = px.bar(fail_counts, x="Component", y="Count", title="Failure Count per Component")
        st.plotly_chart(fig_fail, use_container_width=True)
