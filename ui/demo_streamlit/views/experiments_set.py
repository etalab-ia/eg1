import json
import re
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
from utils import fetch


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
    exp = fetch("get", f"/experiment/{exp_id}", {"with_dataset": "true"})
    if not exp:
        return None

    df = pd.read_json(StringIO(exp["dataset"]["df"]))

    if "answers" in exp:
        answers = {answer["num_line"]: answer["answer"] for answer in exp["answers"]}
        df["answer"] = df.index.map(answers)

    if "results" in exp:
        for result in exp["results"]:
            metric_name = result["metric_name"]
            observations = {obs["num_line"]: obs["score"] for obs in result["observation_table"]}
            df[f"result_{metric_name}"] = df.index.map(observations)

    return df


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


def display_experiment_set_overview(expset, experiments_df):
    """
    returns a dataframe with the list of Experiments and the associated status
    """
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


def display_experiment_details(experimentset, experiments_df):
    experiment_ids = experiments_df["Id"].tolist()
    selected_exp_id = st.selectbox("Select Experiment ID", experiment_ids)
    experiment = next(
        (exp for exp in experimentset.get("experiments", []) if exp["id"] == selected_exp_id), None
    )
    if experiment:
        df_with_results = _get_experiment_data(experiment["id"])
        expe_name = experiment["name"]
        readme = experiment["readme"]
        dataset_name = experiment["dataset"]["name"]
        model_name = experiment.get("model") or "Unknown Model"

        if df_with_results is not None:
            st.write(f"**experiment_id** n° {selected_exp_id}")
            st.write(f"**Name:** {expe_name}")
            st.write(f"**Readme:** {readme}")
            cols = st.columns(2)
            with cols[0]:
                st.write(f"**Dataset:** {dataset_name}")
            with cols[1]:
                st.write(f"**Model:** {model_name}")
            st.dataframe(
                df_with_results,
                use_container_width=True,
                hide_index=False,
                column_config={"Id": st.column_config.TextColumn(width="small")},
            )
        else:
            st.error("Failed to fetch experiment data")


def _all_equal(lst):
    return all(x == lst[0] for x in lst)


def _remove_commons_items(model_params: list[dict], first=True) -> list[dict]:
    if first:
        model_params = deepcopy(model_params)

    common_keys = set.intersection(*(set(d.keys()) for d in model_params))
    for k in common_keys:
        if _all_equal([d[k] for d in model_params]):
            _ = [d.pop(k) for d in model_params]
        elif all(isinstance(d[k], dict) for d in model_params):
            # improves: works with any instead of all
            # take all dict value (recurse)
            # reinsert dict value in same order
            x = [(i, d[k]) for i, d in enumerate(model_params) if isinstance(d[k], dict)]
            idx, params = zip(*x)
            params = _remove_commons_items(list(params), first=False)
            for i, _id in enumerate(idx):
                if not params[i]:
                    model_params[_id].pop(k)
                model_params[_id][k] = params[i]
        elif all(isinstance(d[k], list) for d in model_params):
            # @improves: works with any instead of all
            # take all dict value in  list value (recurse)
            # reinsert dict value in same order
            pass

    return model_params


def _rename_model_variants(experiments: list) -> list:
    """
    Inplace add a _name attribute to experiment several model name are equal to help
    distinguish them
    """
    names = [exp["model"]["name"] for exp in experiments if exp.get("model")]
    if len(set(names)) == len(names):
        return experiments

    names = []
    for i, exp in enumerate(experiments):
        if not exp.get("model"):
            continue

        name = exp["model"]["name"]
        _name = name
        suffix = ""
        if re.search(r"__\d+$", name):
            parts = name.rsplit("__", 1)
            _name = parts[0]
            suffix = parts[1]

        names.append(
            {
                "pos": i,
                "name": name,
                "_name": _name,
                "suffix": suffix,
            }
        )

    # Find the experiments that have an equal _model name
    model_names = defaultdict(list)
    for item in names:
        if not item:
            continue
        model_names[item["_name"]].append(item["pos"])

    # Canonize model names
    for _name, ids in model_names.items():
        if len(ids) <= 1:
            continue

        # List of model params
        model_params = [
            (experiments[i]["model"].get("sampling_params") or {})
            | (experiments[i]["model"].get("extra_params") or {})
            for i in ids
        ]

        # remove commons parameters
        model_diff_params = _remove_commons_items(model_params)

        for model in names:
            pos = next((x for x in ids if model["pos"] == x), None)
            if not pos:
                continue

            # Finally renamed it !
            variant = model_diff_params[ids.index(pos)]
            if variant:
                variant = json.dumps(variant)
                variant = variant.replace('"', "").replace(" ", "")

                experiments[pos]["_model"] = "#".join([_name, variant]) + model["suffix"]


def _find_default_sort_metric(columns):
    """
    find a sensible default metric for sorting results.
    """
    preferred_metrics = ["judge_exactness", "contextual_relevancy"]
    for metric in preferred_metrics:
        if metric in columns:
            return f"{metric}"

    return list(columns)[0] if len(columns) > 0 else None


def _sort_columns(df: pd.DataFrame, first_columns: list) -> pd.DataFrame:
    first_columns = []
    new_column_order = (
        sorted(first_columns)  # Sort the first group of columns
        + sorted([col for col in df.columns if col not in first_columns])  # Sort remaining columns
    )
    return df[new_column_order]


def _check_repeat_mode(experiments: list) -> bool:
    """
    check whether the experiment is related to a repetition
    """
    for exp in experiments:
        name = exp["name"]
        if re.search(r"__\d+$", name):
            return True

    return False


def _format_experimentd_score_df(experiments: list, df: pd.DataFrame):
    experiment_ids = [exp["id"] for exp in experiments]
    experiment_names = [exp["name"] for exp in experiments]
    is_repeat_mode = _check_repeat_mode(experiments)

    if is_repeat_mode and df["model"].notna().all():
        # Lost repetition trailing code.
        df["model"] = df["model"].str.replace(r"__\d+$", "", regex=True)
        # Group by 'model' and calculate mean and std for all numeric columns
        grouped = df.groupby("model").agg(["mean", "std"]).reset_index()

        # Create a new DataFrame to store the results
        result = pd.DataFrame()
        result["model"] = grouped["model"]

        # Iterate over each column (except 'model') to format mean ± std
        for column in df.columns:
            if column not in ["model"]:
                # Format the score as "mean ± std"
                result[column] = (
                    grouped[(column, "mean")].round(2).astype(str)
                    + " ± "
                    + grouped[(column, "std")].round(2).astype(str)
                )

        df = result
    else:
        df["Id"] = experiment_ids
        df["Name"] = experiment_names
        df = df[["Id", "Name"] + [col for col in df.columns if col not in ["Id", "Name"]]]

    default_sort_metric = _find_default_sort_metric(df.columns)
    if default_sort_metric in df.columns:
        df = df.sort_values(by=f"{default_sort_metric}", ascending=False)

    return df


def display_experiment_set_score(experimentset, experiments_df):
    """
    process experiment results dynamically across different experiment types.
    """

    rows = []
    rows_support = []
    experiments = experimentset.get("experiments", [])
    _rename_model_variants(experiments)
    size = experiments[0]["dataset"]["size"]

    for exp in experiments:
        row = {}
        row_support = {}
        if exp.get("_model") or exp.get("model"):
            row["model"] = exp.get("_model") or exp["model"]["name"]
            row_support["model"] = exp.get("_model") or exp["model"]["name"]

        exp = fetch("get", f"/experiment/{exp['id']}?with_results=true")
        if not exp:
            continue

        for metric_results in exp.get("results", []):
            metric = metric_results["metric_name"]
            scores = [
                x["score"] for x in metric_results["observation_table"] if pd.notna(x.get("score"))
            ]
            if scores:
                row[f"{metric}"] = np.mean(scores)
                row_support[f"{metric}_support"] = len(scores)

        rows.append(row)
        rows_support.append(row_support)

    if not rows:
        st.error("No valid experiment results found")
        return

    df = pd.DataFrame(rows)
    df = _sort_columns(df, [])
    df = _format_experimentd_score_df(experiments, df)

    df_support = pd.DataFrame(rows_support)
    df_support = _sort_columns(df_support, [])
    df_support = _format_experimentd_score_df(experiments, df_support)

    st.write("**Score:** Averaged score on experiments metrics")
    if _check_repeat_mode(experiments):
        st.warning("Score are aggregated on model repetition.")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={"Id": st.column_config.TextColumn(width="small")},
    )

    st.write("---")
    st.write(f"**Support:** the number of item on wich the metrics is computed (size = {size})")
    st.dataframe(
        df_support,
        use_container_width=True,
        hide_index=True,
        column_config={"Id": st.column_config.TextColumn(width="small")},
    )


def main():
    experiment_sets = fetch("get", "/experiment_sets")

    expid = st.query_params.get("expset")
    if expid:
        # st.session_state["experimentset"] = next((x for x in experiment_sets if x["id"] == expid), None)
        experimentset = fetch("get", f"/experiment_set/{expid}")
        if not experimentset:
            raise ValueError("experimentset not found: %s" % expid)
        st.session_state["experimentset"] = experimentset

    if st.session_state.get("experimentset"):
        # Get the expet
        experimentset = st.session_state["experimentset"]
        st.query_params.expset = experimentset["id"]

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
                st.query_params.pop("expset")
                st.rerun()

        with col2:
            if st.button("🔄 Refresh Data"):
                expid = experimentset["id"]
                experimentset = fetch("get", f"/experiment_set/{expid}")
                if not experimentset:
                    raise ValueError("experimentset not found: %s" % expid)
                st.session_state["experimentset"] = experimentset

        def show_header():
            status, counts = _get_expset_status(experimentset)
            st.markdown(f"## {experimentset['name']} ")
            st.markdown(f"**experiment_set id**: {experimentset['id']}")
            st.markdown(f'**Readme:** {experimentset.get("readme", "No description available")}')

            finished_ratio = int(
                counts["total_observation_successes"] / counts["observation_length"] * 100
            )
            st.markdown(f"Finished: {finished_ratio}%", unsafe_allow_html=True)
            failure_ratio = int(
                (counts["total_observation_tries"] - counts["total_observation_successes"])
                / counts["observation_length"]
                * 100
            )
            if failure_ratio > 0:
                st.markdown(
                    f"Failure: <span style='color:red;'>{failure_ratio}%</span>",
                    unsafe_allow_html=True,
                )

        show_header()

        # Display tabs
        # --
        tab_index = {
            1: {"key": "scores", "title": "Scores", "func": display_experiment_set_score},
            2: {
                "key": "overview",
                "title": "Set Overview",
                "func": display_experiment_set_overview,
            },
            3: {
                "key": "details",
                "title": "Details by experiment id",
                "func": display_experiment_details,
            },
        }
        tab_reverse = {d["key"]: k for k, d in tab_index.items()}
        # @TODO: how to catch the tab click in order to set the current url query to tab key ?

        tab1, tab2, tab3 = st.tabs(
            [
                tab_index[1]["title"],
                tab_index[2]["title"],
                tab_index[3]["title"],
            ]
        )

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
            tab_index[1]["func"](experimentset, experiments_df)
        with tab2:
            tab_index[2]["func"](experimentset, experiments_df)
        with tab3:
            tab_index[3]["func"](experimentset, experiments_df)

    else:
        st.title("Experiments (Set)")
        if experiment_sets:
            display_experiment_sets(experiment_sets)


main()
