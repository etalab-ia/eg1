import streamlit as st
import pandas as pd
import plotly.express as px
from utils import fetch
from collections import defaultdict

def main():
    st.title("Ops Analysis Dashboard")
    st.write("")

    # Récupération des métriques OPS
    ops_metrics = fetch("get", "/ops_metrics")

    # Affichage des métriques principales
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Experiment Sets", ops_metrics["experiment_sets"])
    col2.metric("Unique Experiments", ops_metrics["unique_experiments"])
    col3.metric("Unique Answers", ops_metrics["unique_answers"])
    col4.metric("Unique Metrics", ops_metrics["unique_metrics"])
    col5.metric("Unique Observations", ops_metrics["unique_observations"])

    # Analyse des modèles évalués par fournisseur
    models_by_provider = defaultdict(list)
    for model in ops_metrics["distinct_models"]:
        model_name = model.get("aliased_name") or model.get("name", "Unknown model")
        provider = model_name.split("/")[0] if "/" in model_name else "hybride"
        models_by_provider[provider].append(model_name)

    # Affichage des modèles évalués
    st.subheader("📈 Models Evaluated")
    for provider, models in models_by_provider.items():
        st.write(f"**{provider}**")
        for model in sorted(models):
            st.write(f"- {model}")
        st.write("---")

main()
