import streamlit as st
import pandas as pd
import numpy as np
from utils import fetch

def fetch_all_experiments():
    endpoint = "/experiments"
    return fetch("get", endpoint)

def fetch_experiment_results(exp_id):
    endpoint = f"/experiment/{exp_id}"
    params = {"with_results": "true"}
    return fetch("get", endpoint, params)

def process_experiment_results(experiment):
    results = experiment.get("results", [])
    df_metrics = {}
    
    for metric_results in results:
        metric_name = metric_results["metric_name"]
        arr = np.array([x["score"] for x in metric_results["observation_table"] if pd.notna(x["score"])])
        
        if len(arr) > 0:
            df = pd.DataFrame([[
                np.mean(arr),
                np.std(arr),
                np.median(arr),
                f"{arr.mean():.2f} ± {arr.std():.2f}",
                len(arr),
            ]], columns=["mean", "std", "median", "mean_std", "support"])
            
            df_metrics[metric_name] = df
    
    return pd.DataFrame({metric_name: df["mean_std"].iloc[0] for metric_name, df in sorted(df_metrics.items())}, index=[experiment["name"]])


def display_all_experiments():
    experiments = fetch_all_experiments()
    
    if not experiments:
        st.error("No experiments found.")
        return
    
    formatted_experiments = []
    
    for exp in experiments:
        if exp["experiment_status"] == "finished" and exp["experiment_set_id"] is None :
            formatted_exp = {
                "id": exp["id"],
                "name": exp["name"],
                "dataset": exp["dataset"]["name"],
                "model": exp["model"]["name"] if exp["model"] else "N/A"
            }
            
            for result in exp.get("results", []):
                metric_name = result["metric_name"]
                scores = [obs["score"] for obs in result["observation_table"] if obs["score"] is not None]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    formatted_exp[f"{metric_name}_score"] = f"{avg_score:.2f}"
            
            formatted_experiments.append(formatted_exp)
    
    df = pd.DataFrame(formatted_experiments)

    metric_columns = [col for col in df.columns if col.endswith("_score")]
    df = df[df[metric_columns].notna().any(axis=1)]

    st.dataframe(df)
    
    if not df.empty:
        selected_exp_id = st.selectbox("Select a finished experiment to view details:", df["id"].tolist())
        
        if st.button("Show Selected Experiment Results"):
            display_experiment_results(selected_exp_id)
    else:
        st.info("No finished experiments found.")


def display_experiment_results(exp_id):
    experiment = fetch_experiment_results(exp_id)
    
    if not experiment:
        return
    
    if experiment["experiment_status"] != "finished":
        st.warning(f"Experiment {exp_id} is not finished yet...")
    
    results_df = process_experiment_results(experiment)
    
    if not results_df.empty:
        st.dataframe(results_df)
    else:
        st.info("No results available for this experiment.")

def main():
    st.title("Experiments (not in a Set)")
    st.info("Here, you can see the experiments that are not in evaluation sets. ")

    options_button = ["View All Experiments (finished)", "View Experiment by ID"]
    view_option = st.radio("Select View Option", options_button)
    
    if view_option == "View All Experiments (finished)":
        display_all_experiments()
    else:
        exp_id = st.number_input("Enter Experiment ID", min_value=1, step=1)
        if st.button("Show Results"):
            display_experiment_results(exp_id)


main()