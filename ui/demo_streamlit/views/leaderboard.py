import pandas as pd
import streamlit as st
from utils import fetch
from operator import itemgetter
from itertools import groupby

@st.cache_data(ttl=600)
def fetch_leaderboard(metric_name="judge_notator", limit=30):
    endpoint = "/leaderboard"
    params = {"metric_name": metric_name, "limit": limit}
    return fetch("get", endpoint, params)

@st.cache_data(ttl=3600)
def fetch_metrics():
    return fetch("get", "/metrics")

def main():
    st.title("EG1 Leaderboard")
    
    metrics = fetch_metrics()
    if not metrics:
        st.error("Unable to fetch metrics. Please try again later.")
        return

    filtered_metrics = [metric for metric in metrics if metric["type"] != "dataset"]
    sorted_metrics = sorted(filtered_metrics, key=itemgetter("type"))
    grouped_metrics = {k: list(v) for k, v in groupby(sorted_metrics, key=itemgetter("type"))}
    
    available_metrics = [metric["name"] for metrics in grouped_metrics.values() for metric in metrics]
    
    metric_name = st.selectbox("Select a metric for the leaderboard", available_metrics)
    
    limit = st.slider("Number of top results to display", min_value=5, max_value=100, value=30, step=5)
    
    st.write(f"## 🏆 Top Performing LLMs Leaderboard - {metric_name.replace('_', ' ').title()}")
    st.write("")

    leaderboard_data = fetch_leaderboard(metric_name, limit)

    if leaderboard_data and leaderboard_data["entries"]:
        df = pd.DataFrame([
            {
                "Model": entry["model_name"],
                "Sampling_param": entry["sampling_param"],
                "Extra_param": entry["extra_param"],
                "Dataset": entry["dataset_name"],
                f"{metric_name.replace('_', ' ').title()} Score": entry["main_metric_score"],
                **entry["other_metrics"]
            }
            for entry in leaderboard_data["entries"]
        ])

        st.dataframe(
            df.sort_values(f"{metric_name.replace('_', ' ').title()} Score", ascending=False, na_position='last'),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write("No data available for the leaderboard.")

main()
