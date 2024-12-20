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


def get_experiment_data(exp_id):
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
    status, counts = _get_expset_status(expset)
    st.write(f"## Overview of experiment set: {expset['name']}")
    st.write(f"experiment_set id: {expset['id']}")
    finished_ratio = int(counts["total_observation_successes"] / counts["observation_length"] * 100)
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


def display_experiment_set_result(expset, experiments_df):
    st.write("## Results of the Experiment Set")

    total_experiments = len(experiments_df)
    total_success = experiments_df["Num success"].sum()

    st.write(f"Total Experiments: {total_experiments}")
    st.write(f"Total Successful Experiments: {total_success}")


def display_experiment_sets(experiment_sets):
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
        df_with_results, dataset_name, model_name = get_experiment_data(selected_exp_id)
        if df_with_results is not None:
            st.write(f"### Detailed results of the experiment id={selected_exp_id} ")
            st.write(f"**Dataset:** {dataset_name}")
            st.write(f"**Model:** {model_name}")
            st.dataframe(df_with_results)
        else:
            st.error("Failed to fetch experiment data")


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
