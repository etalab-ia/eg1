import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
from utils import fetch
from io import StringIO


def _get_expset_status(expset: dict) -> tuple[dict, dict]:
    status_codes = {
        "pending": {"text": "Experiments did not start yet", "color": "yellow"},
        "running": {"text": "Experiments are running", "color": "orange"},
        "finished": {"text": "All experiments are finished", "color": "green"},
    }

    counts = dict(
        total_answer_tries=sum(exp["num_try"] for exp in expset["experiments"]),
        total_answer_successes=sum(exp["num_success"] for exp in expset["experiments"]),
        total_observation_tries=sum(exp["num_observation_try"] for exp in expset["experiments"]),
        total_observation_successes=sum(exp["num_observation_success"] for exp in expset["experiments"]),
        answer_length=sum(exp["dataset"]["size"] for exp in expset["experiments"]),
        observation_length=sum(exp["dataset"]["size"]*exp["num_metrics"] for exp in expset["experiments"]),
    )  # fmt: skip

    # Running status
    if all(exp["experiment_status"] == "pending" for exp in expset["experiments"]):
        status = status_codes["pending"]
    elif all(exp["experiment_status"] == "finished" for exp in expset["experiments"]):
        status = status_codes["finished"]
    else:
        status = status_codes["running"]

    return status, counts


def _get_experiment_data(exp_id):
    """
    for each exp_id, returns query, answer true, answer llm and metrics
    """
    response = fetch("get", f"/experiment/{exp_id}", {"with_dataset": "true"})
    if not response:
        return None

    df = pd.read_json(StringIO(response["dataset"]["df"]))

    if "answers" in response:
        answers = {answer["num_line"]: answer["answer"] for answer in response["answers"]}
        df["answer"] = df.index.map(answers)

    if "results" in response:
        for result in response["results"]:
            metric_name = result["metric_name"]
            observations = {obs["num_line"]: obs["score"] for obs in result["observation_table"]}
            df[f"result_{metric_name}"] = df.index.map(observations)

    dataset_name = response.get("dataset", {}).get("name", "Unknown Dataset")
    model_name = response.get("model", {}).get("name", "Unknown Model")

    return df, dataset_name, model_name


def display_experiment_set_overview(expset, experiments_df):
    """
    returns a dataframe with the list of Experiments and the associated status
    """

    status, counts = _get_expset_status(expset)
    st.markdown(f"## Overview of experiment set: ~~ {expset['name']} ~~")
    st.markdown(f"experiment_set id: {expset['id']}")
    finished_ratio = int(counts["total_observation_successes"] // counts["observation_length"] * 100)
    st.markdown(f"Finished: {finished_ratio}%", unsafe_allow_html=True)
    failure_ratio = int(
        (counts["total_observation_tries"] - counts["total_observation_successes"])
        / counts["observation_length"]
        * 100
    )
    if failure_ratio > 0:
        st.markdown(
            f"Failure: <span style='color:red;'>{failure_ratio}%</span>", unsafe_allow_html=True
        )
        
 
    st.markdown(expset.get("readme", "No description available"))

    row_height = 35
    header_height = 35
    border_padding = 5
    dynamic_height = len(experiments_df) * row_height + header_height + border_padding

    st.dataframe(
        experiments_df,
        use_container_width=True,
        hide_index=True,
        height=dynamic_height,
        column_config={"Id": st.column_config.TextColumn(width="small")},
    )


def display_experiment_sets(experiment_sets):
    """
    returns the list of experiments set, with their status/info
    """
    cols = st.columns(3)

    for idx, exp_set in enumerate(experiment_sets):
        status, counts = _get_expset_status(exp_set)

        # Failure status
        has_failure = False
        if counts["total_observation_tries"] > counts["total_observation_successes"]:
            has_failure = True

        status_description = status["text"]
        status_color = status["color"]
        if has_failure:
            status_description += " with some failure"
            status_color = f"linear-gradient(to right, {status_color} 50%, red 50%)"

        when = datetime.fromisoformat(exp_set["created_at"]).strftime("%d %B %Y")
        with cols[idx % 3]:
            with st.container(border=True):
                st.markdown(
                    f"<div style='position: absolute; top: 10px; right: 10px; "
                    f"width: 10px; height: 10px; border-radius: 50%; "
                    f"background: {status_color};' "
                    f"title='{status_description}'></div>",
                    unsafe_allow_html=True,
                )

                if st.button(f"{exp_set['name']}", key=f"exp_set_{idx}"):
                    st.session_state["experimentset"] = exp_set
                    st.rerun()

                st.markdown(exp_set.get("readme", "No description available"))

                col1, col2, col3 = st.columns([1 / 6, 2 / 6, 3 / 6])
                with col1:
                    st.caption(f'id: {exp_set["id"]} ')
                with col2:
                    st.caption(f'Experiments: {len(exp_set["experiments"])} ')
                with col3:
                    st.caption(f"Created on {when}")

                if has_failure:
                    with st.expander("Failure Analysis", expanded=False):
                        for exp in exp_set["experiments"]:
                            if exp["num_try"] != exp["num_success"]:
                                st.write(
                                    f"id: {exp['id']} name: {exp['name']} (failed on output generation)"
                                )
                                continue

                            if exp["num_observation_try"] != exp["num_observation_success"]:
                                st.write(
                                    f"id: {exp['id']} name: {exp['name']} (failed on score computation)"
                                )
                                continue


def display_experiment_details(experimentset, experiments_df):
    experiment_ids = experiments_df["Id"].tolist()
    selected_exp_id = st.selectbox("Select Experiment ID", experiment_ids)
    if selected_exp_id:
        df_with_results, dataset_name, model_name = _get_experiment_data(selected_exp_id)
        if df_with_results is not None:
            cols = st.columns(4)
            with cols[0]:
                st.write(f"**experiment_id** n° {selected_exp_id}")
            with cols[1]:
                st.write(f"**Dataset:** {dataset_name}")
            with cols[2]:
                st.write(f"**Model:** {model_name}")
            st.dataframe(df_with_results)
        else:
            st.error("Failed to fetch experiment data")


def process_experiment_results(experimentset):
    """
    process experiment results dynamically across different experiment types.
    """
    rows = []
    metrics = set()
    experiment_names = [exp["name"] for exp in experimentset.get("experiments", [])]
    
    is_repeat_mode = _check_repeat_mode(experiment_names)

    for exp in experimentset.get("experiments", []):
        if exp["experiment_status"] != "finished":
            st.warning(f"Warning: experiment {exp['id']} is not finished yet...")
            continue

        response = fetch("get", f"/experiment/{exp['id']}?with_results=true")
        if not response:
            continue

        model_name = response["model"]["name"]
        extra_params = response["model"].get("extra_params", {})
        variant = _extract_experiment_variant(extra_params)
        row = {"model": f"{model_name}_{variant}" if variant else model_name}

        for metric_results in response.get("results", []):
            metric = metric_results["metric_name"]
            metrics.add(metric)
            scores = [x["score"] for x in metric_results["observation_table"] if pd.notna(x.get("score"))]
            if scores:
                row[f"{metric}_mean"] = np.mean(scores)
                row[f"{metric}_std"] = np.std(scores)
                row[f"{metric}_support"] = len(scores)

        rows.append(row)

    if not rows:
        st.error("No valid experiment results found")
        return None

    df = pd.DataFrame(rows)
    
    if is_repeat_mode:
        df = df.groupby("model").agg({col: ['mean', 'std'] for col in df.columns if col.endswith('_mean')})
        df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
    else:
        default_sort_metric = _find_default_sort_metric(metrics)
        if default_sort_metric and f"{default_sort_metric}_mean" in df.columns:
            df = df.sort_values(by=f"{default_sort_metric}_mean", ascending=False)

    return df

def _check_repeat_mode(experiment_names):
    """
    check whether the experiment is related to a repetition
    """
    if len(experiment_names) <= 1:
        return False
    
    base_names = [name.rsplit('_', 1)[0] for name in experiment_names]
    
    if len(set(base_names)) == 1:
        suffixes = [name.split('_')[-1] for name in experiment_names]
        return all(suffix.isdigit() for suffix in suffixes)
    
    return False

def _extract_experiment_variant(extra_params: dict):
    """
    extract a meaningful variant identifier from extra parameters.
    """
    if not extra_params:
        return ""
    
    if "rag" in extra_params:
        if "limit" in extra_params["rag"]:
            return f"limit_{extra_params['rag']['limit']}"
    
    return str(list(extra_params.keys())[0]) if extra_params else ""

def _find_default_sort_metric(metrics):
    """
    find a sensible default metric for sorting results.
    """
    preferred_metrics = ['judge_exactness', 'contextual_relevancy']
    for metric in preferred_metrics:
        if metric in metrics:
            return f"{metric}_mean"
    
    return list(metrics)[0] + "_mean" if metrics else None

def display_experiment_results(experimentset):
    results_df = process_experiment_results(experimentset)
    
    if results_df is not None:
        st.write("### Experiment Results")
        st.dataframe(results_df)

def display_experiment_set_result(experimentset, experiments_df):
    st.write("## Results of the Experiment Set")

    total_experiments = len(experiments_df)
    all_successful = (experiments_df["Num try"] == experiments_df["Num success"]).all()

    if all_successful:
        display_experiment_results(experimentset)
    else:
        st.error("Detailed results cannot be displayed as not all experiments are successful")
        cols = st.columns(6)
        with cols[0]:
            st.write(f"Total Experiments: {total_experiments}")
        with cols[1]:
            st.write(f"Failure Experiments: {total_experiments - (experiments_df['Num success'] > 0).sum()}")


def main():
    if st.session_state.get("experimentset"):
        # Get the expet
        experimentset = st.session_state["experimentset"]

        # Build the expset dataframe
        experiments_df = pd.DataFrame(
            [
                {
                    "Id": exp["id"],
                    "Name": exp["name"],
                    "Status": exp["experiment_status"],
                    "Created at": exp["created_at"],
                    "Num try": exp["num_try"],
                    "Num success": exp["num_success"],
                    "Num observation try": exp["num_observation_try"],
                    "Num observation success": exp["num_observation_success"],
                }
                for exp in experimentset.get("experiments", [])
            ]
        )
        experiments_df.sort_values(by="Id", ascending=True, inplace=True)

        # Horizontal menu toolbar
        # --
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button(":arrow_left: Go back", key="go_back"):
                st.session_state["experimentset"] = None
                st.rerun()

        with col2:
            if st.button("🔄 Refresh Data"):
                expid = experimentset["id"]
                experimentset = fetch("get", f"/experiment_set/{expid}")
                if not experimentset:
                    raise ValueError("experimentset not found: %s" % expid)
                st.session_state["experimentset"] = experimentset

        # Display tabs
        # --
        tab1, tab2, tab3 = st.tabs(["Set Overview", "Results", "Detail by experiment id"])

        def show_warning_in_tabs(message):
            with tab1:
                st.warning(message)
            with tab2:
                st.warning(message)
            with tab3:
                st.warning(message)

        df = experiments_df  # alias
        if not (df["Status"] == "finished").all():
            show_warning_in_tabs("Warning: some experiments are not finished.")
        if df["Num success"].sum() != df["Num try"].sum():
            show_warning_in_tabs("Warning: some experiments are failed.")
        if df["Num observation success"].sum() != df["Num observation try"].sum():
            show_warning_in_tabs("Warning: some metrics are failed.")

        with tab1:
            display_experiment_set_overview(experimentset, experiments_df)
        with tab2:
            display_experiment_set_result(experimentset, experiments_df)
        with tab3:
            display_experiment_details(experimentset, experiments_df)

    else:
        st.title("Experiments (Set)")
        experiment_sets = fetch("get", "/experiment_sets")
        if experiment_sets:
            display_experiment_sets(experiment_sets)


main()
