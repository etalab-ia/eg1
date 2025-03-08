import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import fetch
from collections import defaultdict
from datetime import datetime
from io import StringIO


def create_report(experiment_sets):
    nb_expSet = len(experiment_sets)
    
    data = {
        "unique_exp_ids": set(),
        "unique_model_name": set(),
        "unique_answer_ids": set(),
        "unique_metric_ids": set(),
        "unique_observation_ids": set(),
        "model_providers": defaultdict(int)
    }
    
    for experiment in experiment_sets:
        for exp in experiment["experiments"]:
            data["unique_exp_ids"].add(exp["id"])
            model_name = exp.get("model", {}).get("name") or exp.get("model", {}).get("aliased_name", "Unknown Model")
            data["unique_model_name"].add(model_name)
            
            # Provider
            if "/" in model_name:
                provider = model_name.split("/")[0]
            else:
                provider = "hybride"
            data["model_providers"][provider] += 1
            
            _experiment_detail = fetch("get", f"/experiment/{exp['id']}", {"with_dataset": "true"})
            
            if "answers" in _experiment_detail:
                for answer in _experiment_detail["answers"]:
                    data["unique_answer_ids"].add(answer.get("id"))
            
            if "results" in _experiment_detail:
                for result in _experiment_detail["results"]:
                    data["unique_metric_ids"].add(result.get("id"))
                    if "observation_table" in result:
                        for observation in result["observation_table"]:
                            data["unique_observation_ids"].add(observation.get("id"))

    st.title("Ops Analysis Dashboard")
    st.write("")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Experiment Sets", nb_expSet)
    col2.metric("Unique Experiments", len(data["unique_exp_ids"]))
    col3.metric("Unique Answers", len(data["unique_answer_ids"]))
    col4.metric("Unique Metrics", len(data["unique_metric_ids"]))
    col5.metric("Unique Observations", len(data["unique_observation_ids"]))

    st.subheader("📈 Models Evaluated")
    models_by_provider = defaultdict(list)
    for model in sorted(list(data["unique_model_name"])):
        provider = model.split("/")[0] if "/" in model else "hybride"
        models_by_provider[provider].append(model)

    for provider, models in models_by_provider.items():
        st.write(f"**{provider}**")
        for model in models:
            st.write(f"- {model}")
        st.write("---")


def main():

    experiment_sets = fetch("get", "/experiment_sets")

    if not experiment_sets:
        st.warning("⚠️ No experiment sets found.")
        return

    create_report(experiment_sets)

main()
