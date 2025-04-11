import json
import re
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import fetch, hash_string


#
# Cached method for critical data fetching
#


def _fetch(method, endpoint, data=None, refresh=False):
    if refresh:
        __fetch.clear(method, endpoint, data)

    return __fetch(method, endpoint, data)


@st.cache_data(ttl=600, max_entries=10)
def __fetch(method, endpoint, data=None):
    return fetch(method, endpoint, data)


def _fetch_experimentset(expid, partial_expset, refresh=False):
    if refresh:
        __fetch_experimentset.clear(expid, partial_expset)

    return __fetch_experimentset(expid, partial_expset)


@st.cache_data(ttl=600, max_entries=3)
def __fetch_experimentset(expid, partial_expset, refresh=False):
    if refresh:
        _fetch_experimentset.clear(expid, partial_expset)

    experimentset = partial_expset
    if not experimentset:
        raise ValueError("experimentset not found: %s" % expid)

    # Fetch experiment results
    for i, expe in enumerate(experimentset["experiments"]) or []:
        expe = fetch("get", f"/experiment/{expe['id']}", {"with_results": True})
        if not expe:
            continue
        experimentset["experiments"][i] = expe

    return experimentset


#
# @TODO: codefactor and triage
#


def _get_expset_status(expset: dict) -> tuple[dict, dict]:
    status_codes = {
        "pending": {"text": "Experiments did not start yet", "color": "yellow"},
        "running": {"text": "Experiments are running", "color": "orange"},
        "finished": {"text": "All experiments are finished", "color": "green"},
    }

    counts = dict(
        total_answer_tries=sum(expe["num_try"] for expe in expset["experiments"]),
        total_answer_successes=sum(expe["num_success"] for expe in expset["experiments"]),
        total_observation_tries=sum(expe["num_observation_try"] for expe in expset["experiments"]),
        total_observation_successes=sum(expe["num_observation_success"] for expe in expset["experiments"]),
        answer_length=sum(expe["dataset"]["size"] for expe in expset["experiments"]),
        observation_length=sum(expe["dataset"]["size"]*expe["num_metrics"] for expe in expset["experiments"]),
    )  # fmt: skip

    # Running status
    if all(expe["experiment_status"] == "pending" for expe in expset["experiments"]):
        status = status_codes["pending"]
    elif all(expe["experiment_status"] == "finished" for expe in expset["experiments"]):
        status = status_codes["finished"]
    else:
        status = status_codes["running"]

    return status, counts


def _get_experiment_data(exp_id):
    """
    for each exp_id, returns query, answer true, answer llm and metrics
    """
    expe = _fetch(
        "get",
        f"/experiment/{exp_id}",
        {"with_dataset": True, "with_results": True},
        refresh=st.session_state.get("refresh_experimentset"),
    )
    if not expe:
        return None

    df = pd.read_json(StringIO(expe["dataset"]["df"]))

    # Merge answers and metrics into the dataset dataframe
    if "answers" in expe:
        answers = {answer["num_line"]: answer["answer"] for answer in expe["answers"]}
        errors = {answer["num_line"]: answer["error_msg"] for answer in expe["answers"]}
        df["answer"] = df.index.map(answers)
        if any(error is not None for error in errors.values()):
            df["answer_errors"] = df.index.map(errors)

    if "results" in expe:
        for result in expe["results"]:
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

                if st.button(f"{exp_set['name']}", key=f"pick_expe_{idx}"):
                    st.query_params["expset"] = exp_set["id"]
                    st.rerun()

                st.markdown(exp_set.get("readme", "No description available"))

                col1, col2, col3 = st.columns([1 / 6, 2 / 6, 3 / 6])
                with col1:
                    st.caption(f"id: {exp_set['id']} ")
                with col2:
                    st.caption(f"Experiments: {len(exp_set['experiments'])} ")
                with col3:
                    st.caption(f"Created the {when}")

    # Show orphan experiments
    # --
    st.markdown("---")
    with st.container(border=True):
        st.markdown(
            f"<div style='position: absolute; top: 10px; right: 10px; "
            f"width: 10px; height: 10px; border-radius: 50%; "
            f"background: {status_color};' "
            f"title='{status_description}'></div>",
            unsafe_allow_html=True,
        )

        if st.button("Orphan experiments", key="pick_expe_orphan"):
            st.query_params["expset"] = "orphan"
            st.rerun()

        st.markdown("The experiments that are not in evaluation sets.")


def display_experiment_set_overview(experimentset, experiments_df):
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
        (expe for expe in experimentset.get("experiments", []) if expe["id"] == selected_exp_id),
        None,
    )
    if experiment:
        full_df = _get_experiment_data(experiment["id"])

        if full_df is not None:
            st.write(f"**experiment_id** n° {selected_exp_id}")
            st.write(f"**Name:** {experiment['name']}")
            st.write(f"**Readme:** {experiment['readme']}")

            cols = st.columns(2)
            with cols[0]:
                st.write(f"**Dataset:** {experiment['dataset']['name']}")
                st.write(f"**Judge model:** {experiment['judge_model']}")
            with cols[1]:
                model_name = experiment.get("model") or "Undefined Model"
                st.write(f"**Model:** {model_name}")

            st.dataframe(
                full_df,
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
    names = [expe["model"]["name"] for expe in experiments if expe.get("model")]
    if len(set(names)) == len(names):
        return experiments

    names = []
    for i, expe in enumerate(experiments):
        if not expe.get("model"):
            continue

        name = expe["model"]["name"]
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
            (experiments[id]["model"].get("sampling_params") or {})
            | (experiments[id]["model"].get("extra_params") or {})
            for id in ids
        ]

        # Manage prompt_system param by computing its hash
        for i, id in enumerate(ids):
            if not experiments[id]["model"].get("prompt_system"):
                continue
            model_params[i]["sys_prompt"] = hash_string(experiments[id]["model"]["prompt_system"], 4)

        # remove commons parameters
        model_diff_params = _remove_commons_items(model_params)

        for model in names:
            pos = next((x for x in ids if model["pos"] == x), None)
            if pos is None:
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
    preferred_metrics = ["answer_relevancy", "judge_notator", "judge_exactness"]
    for metric in preferred_metrics:
        if metric in columns:
            return f"{metric}"

    return list(columns)[0] if len(columns) > 0 else None


def _extract_mean(value):
    try:
        return float(value.split("±")[0].strip())
    except:
        return value  # Return original value if not in expected format


def _sort_score_df(*dfs, reset_index=False):
    if len(dfs) == 0:
        return

    df = dfs[0]
    sorting_metric = _find_default_sort_metric(df.columns)
    df.sort_values(
        by=sorting_metric,
        key=lambda x: x.map(_extract_mean),
        ascending=False,
        inplace=True,
    )
    # Store the sorted index before resetting it
    sorted_idx = df.index.copy()
    for df in dfs:
        # Reorder df2 inplace to match df1's order
        df.loc[:] = df.loc[sorted_idx].values
        if reset_index:
            # Reset indices inplace
            df.reset_index(drop=True, inplace=True)


def _sort_columns(df: pd.DataFrame, first_columns: list) -> pd.DataFrame:
    first_columns = []
    new_column_order = sorted(first_columns) + sorted(  # Sort the first group of columns
        [col for col in df.columns if col not in first_columns]
    )  # Sort remaining columns
    return df[new_column_order]


def _check_repeat_mode(experiments: list) -> bool:
    """
    check whether the experiment is related to a repetition
    """
    for expe in experiments:
        name = expe["name"]
        if re.search(r"__\d+$", name):
            return True

    return False


def _format_model_params(expe):
    if not expe.get("model"):
        return None

    model = expe["model"].copy()
    model_params = (model.get("sampling_params") or {}) | (model.get("extra_params") or {})
    if model.get("prompt_system"):
        model_params["sys_prompt"] = hash_string(model["prompt_system"], 4)

    return model_params


def _format_experiments_score_df(experiments: list, df: pd.DataFrame) -> (bool, pd.DataFrame):
    experiment_ids = [expe["id"] for expe in experiments]
    experiment_names = [expe["name"] for expe in experiments]
    is_repeat_mode = _check_repeat_mode(experiments)
    result = None

    if is_repeat_mode and df["model"].notna().all():
        has_repeat = True
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
                mean_ = grouped[(column, "mean")].round(2).astype(str)
                std_ = grouped[(column, "std")].round(2).astype(str)
                if all(x is None or x == 0 or np.isnan(x) for x in std_.astype(float)):
                    result[column] = mean_
                else:
                    result[column] = mean_ + " ± " + std_

    if result is None or len(result) == len(df):
        df["Id"] = experiment_ids
        # df["name"] = experiment_names
        df = df[["Id", "model"] + [col for col in df.columns if col not in ["Id", "model"]]]
        has_repeat = False
    else:
        df = result

    return has_repeat, df


def display_experiment_set_score(experimentset, experiments_df):
    """Affiche les scores de l'ensemble d'expériences."""
    experiments = experimentset.get("experiments", [])
    _rename_model_variants(experiments)
    size = experiments[0]["dataset"]["size"]

    available_judges = sorted(
        list(set(expe.get("judge_model") for expe in experiments if expe.get("judge_model")))
    ) or ["No_judge_found"]

    col1, col2 = st.columns([6, 2])
    with col1:
        st.write("**Score:** Averaged score on experiments metrics")
    with col2:
        st.write(f"**Judge model:** {available_judges[0] if available_judges else 'No judge found'}")

    if len(available_judges) > 1:
        st.warning(f"Multiple judge models found: {', '.join(available_judges)}")

    rows = []
    rows_support = []
    for expe in experiments:
        row = {}
        row_support = {}

        # Determine model name
        if expe.get("_model") or expe.get("model"):
            model_name = expe.get("_model") or expe["model"]["aliased_name"] or expe["model"]["name"]
        else:
            model_name = f"Undefined model ({expe['name']})"
        row["model"] = model_name
        row_support["model"] = model_name

        # Aggregate results/scores
        for metric_results in expe.get("results", []):
            metric = metric_results["metric_name"]
            scores = [x["score"] for x in metric_results["observation_table"] if pd.notna(x.get("score"))]
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

    if "model" not in df.columns:
        df["model"] = [expe.get("name", "Unknown Model") for expe in experiments]

    try:
        has_repeat, df = _format_experiments_score_df(experiments, df)
    except (ValueError, TypeError) as err:
        st.error("No valid result found, try again later...")
        return

    df_support = pd.DataFrame(rows_support)
    df_support = _sort_columns(df_support, [])
    _, df_support = _format_experiments_score_df(experiments, df_support)

    if has_repeat:
        st.warning("Score are aggregated on model repetition.")

    _sort_score_df(df, df_support)

    # To highlight min/max values in each column
    def highlight_min_max(df):
        # Create an empty DataFrame with the same shape as our original
        highlight_df = pd.DataFrame("", index=df.index, columns=df.columns)

        # For each column, find the min and max values and style them
        for col in df.columns:
            if col in ["id", "Id"]:
                continue

            col_means = df[col].apply(_extract_mean)
            if col_means.dtype in [np.float64, np.int64]:
                max_idx = col_means.idxmax()
                min_idx = col_means.idxmin()
                highlight_df.loc[max_idx, col] = "font-weight: bold; color: green"
                highlight_df.loc[min_idx, col] = "font-weight: bold; color: red"

        return highlight_df

    float_columns = df.select_dtypes(include=["float"]).columns
    st.dataframe(
        # Apply styling
        df.style.apply(highlight_min_max, axis=None).format("{:.2f}", subset=float_columns),
        use_container_width=True,
        hide_index=True,
        column_config={"Id": st.column_config.TextColumn(width="small")},
    )

    st.write("---")
    st.write(f"**Support:** the numbers of item on wich the metrics is computed (total items = {size})")
    st.dataframe(
        df_support,
        use_container_width=True,
        hide_index=True,
        column_config={"Id": st.column_config.TextColumn(width="small")},
    )


def report_ops_global(exp_set):
    """
    Generates a DataFrame report with experiment set statuses and displays it,
    along with a bar chart visualization.
    """
    st.subheader("Status by Experiment Set")

    report_data = []
    if exp_set:
        status, counts = _get_expset_status(exp_set)

        has_failure = counts["total_observation_tries"] > counts["total_observation_successes"]
        if has_failure:
            with st.expander("Failure Analysis", expanded=False):
                for expe in exp_set["experiments"]:
                    if expe["num_try"] != expe["num_success"]:
                        st.write(f"id: {expe['id']} name: {expe['name']} (failed on output generation)")
                        continue

                    if expe["num_observation_try"] != expe["num_observation_success"]:
                        st.write(f"id: {expe['id']} name: {expe['name']} (failed on score computation)")
                        continue

        report_data.append(
            {
                "Experiment Set Name": exp_set["name"],
                "Status": status["text"],
                "Total Experiments": len(exp_set["experiments"]),
                "Answer Tries": counts["total_answer_tries"],
                "Answer Successes": counts["total_answer_successes"],
                "Observation Tries": counts["total_observation_tries"],
                "Observation Successes": counts["total_observation_successes"],
                "Has Failure": has_failure,
            }
        )

    report_df = pd.DataFrame(report_data)
    st.dataframe(
        report_df,
        use_container_width=True,
        hide_index=True,
    )


def update_model_data(model_data, experiment):
    # Determine model name
    if experiment.get("model"):
        model_name = experiment["model"]["aliased_name"] or experiment["model"]["name"]
    else:
        model_name = f"Undefined model ({experiment['name']})"

    has_error = any(answer.get("error_msg") for answer in experiment.get("answers", []))

    if has_error:
        model_data[model_name]["failed"] += 1
    else:
        status = experiment["experiment_status"]
        model_data[model_name][status] += 1
        model_data[model_name]["no_failed"] += 1


def update_metric_data(metric_data, experiment):
    has_error = any(answer.get("error_msg") for answer in experiment.get("answers", []))
    if "results" in experiment and not has_error:
        for result in experiment["results"]:
            metric_name = result["metric_name"]
            metric_status = result["metric_status"]
            metric_data[metric_name][metric_status] += 1
            metric_data[metric_name]["no_failed"] += 1
    elif has_error:
        metric_data["Unknown"]["failed"] += 1


def calculate_failure_rate(row):
    total = row["failed"] + row["no_failed"]
    return row["failed"] / total if total > 0 else 0


def report_model_and_metric(experimentset):
    """Analyzes experiment statuses by model and metric, including failed experiments and failure rates."""
    model_data = defaultdict(
        lambda: {
            "finished": 0,
            "running": 0,
            "running_answers": 0,
            "running_metrics": 0,
            "pending": 0,
            "failed": 0,
            "no_failed": 0,
        }
    )
    metric_data = defaultdict(lambda: {"finished": 0, "running": 0, "pending": 0, "failed": 0, "no_failed": 0})

    for experiment in experimentset["experiments"]:
        update_model_data(model_data, experiment)
        update_metric_data(metric_data, experiment)

    model_report = pd.DataFrame.from_dict(model_data, orient="index")
    model_report["Total"] = model_report[["finished", "running", "pending"]].sum(axis=1)
    model_report["Failure Rate"] = model_report.apply(calculate_failure_rate, axis=1)

    columns_order = [
        "finished",
        "running",
        "pending",
        "Total",
        "failed",
        "no_failed",
        "Failure Rate",
    ]
    model_report = model_report[columns_order]
    model_for_graph = model_report.copy()
    model_report.columns = [col.title().replace("_", " ") for col in model_report.columns]

    model_report.columns = pd.MultiIndex.from_tuples(
        [
            ("Status", "Finished"),
            ("Status", "Running"),
            ("Status", "Pending"),
            ("Status", "Total"),
            ("Failure Analysis", "Failed"),
            ("Failure Analysis", "No Failed"),
            ("Failure Analysis", "Failure Rate"),
        ]
    )

    st.subheader("Experiment Status by Model")
    st.dataframe(model_report, use_container_width=True)

    fig_model = px.bar(
        model_for_graph,
        x=model_for_graph.index,
        y=["finished", "running", "pending", "failed"],
        title="Experiment Status by Model",
        labels={"value": "Count", "index": "Model"},
    )
    st.plotly_chart(fig_model, use_container_width=True)

    metric_report = pd.DataFrame.from_dict(metric_data, orient="index")
    metric_report["Total"] = metric_report[["finished", "running", "pending"]].sum(axis=1)
    metric_report["Failure Rate"] = metric_report.apply(calculate_failure_rate, axis=1)

    metric_report = metric_report[columns_order]
    metric_for_graph = metric_report.copy()

    metric_report.columns = [col.title().replace("_", " ") for col in metric_report.columns]

    metric_report.columns = pd.MultiIndex.from_tuples(
        [
            ("Status", "Finished"),
            ("Status", "Running"),
            ("Status", "Pending"),
            ("Status", "Total"),
            ("Failure Analysis", "Failed"),
            ("Failure Analysis", "No Failed"),
            ("Failure Analysis", "Failure Rate"),
        ]
    )

    st.subheader("Experiment Status by Metric")
    st.dataframe(metric_report, use_container_width=True)

    fig_metric = px.bar(
        metric_for_graph,
        x=metric_for_graph.index,
        y=["finished", "running", "pending", "failed"],
        title="Experiment Status by Metric",
        labels={"value": "Count", "index": "Metric"},
    )
    st.plotly_chart(fig_metric, use_container_width=True)


def display_ops_analysis(experimentset):
    report_ops_global(experimentset)
    report_model_and_metric(experimentset)


def show_header(experimentset):
    status, counts = _get_expset_status(experimentset)
    st.markdown(f"## {experimentset['name']}")
    col1, col2 = st.columns([1 / 12, 1])
    with col1:
        st.markdown(f"**Id**: {experimentset['id']}")
    with col2:
        try:
            when = datetime.fromisoformat(experimentset["created_at"]).strftime("%d %B %Y")
        except ValueError:
            when = "N/A"
        st.caption(f"Created the {when}")
    st.markdown(f"**Readme:** {experimentset.get('readme', 'No description available')}")

    finished_ratio = int(counts["total_observation_successes"] / counts["observation_length"] * 100)
    failure_ratio = int(
        (counts["total_observation_tries"] - counts["total_observation_successes"])
        / counts["observation_length"]
        * 100
    )

    run_status = f"**Finished**: {finished_ratio}%"
    if failure_ratio > 0:
        run_status += f" &nbsp;&nbsp;&nbsp; Failure: <span style='color:red;'>{failure_ratio}%</span>"

    st.markdown(run_status, unsafe_allow_html=True)


def main():
    # Fetch or re-fetch data
    # --
    experiment_sets = _fetch("get", "/experiment_sets", refresh=st.session_state.get("refresh_main"))

    # View Branching
    # --
    expid = st.query_params.get("expset") or st.session_state.get("expset_id")
    if expid:
        st.session_state["expset_id"] = expid
        st.query_params.expset = expid

        # Get the expset (or the orphan experiments)
        if expid.isdigit():
            experimentset = next((x for x in experiment_sets if x["id"] == int(expid)), None)
            experimentset = _fetch_experimentset(
                expid,
                experimentset,
                refresh=st.session_state.get("refresh_experimentset"),
            )
        elif expid == "orphan":
            experimentset = {
                "id": None,
                "name": "Orphan experiments",
                "created_at": "",
                "experiments": fetch("get", "/experiments", {"orphan": True, "backward": True}),
            }

        else:
            st.error("Invalid experiment set id: %s" % expid)
            return

        # Horizontal menu toolbar
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(":arrow_left: Go back", key="go_back"):
                st.session_state["expset_id"] = None
                st.query_params.pop("expset")
                st.rerun()

        with col2:
            if st.button("🔄 Refresh", key="refresh_experimentset"):
                st.rerun()

        experiments_df = pd.DataFrame(
            [
                {
                    "Id": expe["id"],
                    "Name": expe["name"],
                    "Dataset": expe["dataset"]["name"],
                    "Model": (expe["model"]["aliased_name"] or expe["model"]["name"])
                    if expe.get("model")
                    else "Undefined model",
                    "Model params": _format_model_params(expe),
                    "Status": expe["experiment_status"],
                    "Created at": expe["created_at"],
                    "Num try": expe["num_try"],
                    "Num success": expe["num_success"],
                    "Num observation try": expe["num_observation_try"],
                    "Num observation success": expe["num_observation_success"],
                }
                for expe in experimentset["experiments"]
            ]
        )
        experiments_df.sort_values(by="Id", ascending=True, inplace=True)

        show_header(experimentset)

        # Display tabs
        # --
        tab_index = {
            1: {
                "key": "scores",
                "title": "⭐ Scores",
                "func": display_experiment_set_score,
            },
            2: {
                "key": "overview",
                "title": "📊 Set Overview",
                "func": display_experiment_set_overview,
            },
            3: {
                "key": "details",
                "title": "📝 Details by Experiment Id",
                "func": display_experiment_details,
            },
            4: {
                "key": "ops",
                "title": "🚨 Ops Analysis",
                "func": display_ops_analysis,
            },
        }
        tab_reverse = {d["key"]: k for k, d in tab_index.items()}
        # @TODO: how to catch the tab click in order to set the current url query to tab key ?

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                tab_index[1]["title"],
                tab_index[2]["title"],
                tab_index[3]["title"],
                tab_index[4]["title"],
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
            show_warning_in_tabs("Warning: some answers are failed.")
        if df["Num observation success"].sum() != df["Num observation try"].sum():
            show_warning_in_tabs("Warning: some metrics are failed.")

        with tab1:
            tab_index[1]["func"](experimentset, experiments_df)
        with tab2:
            tab_index[2]["func"](experimentset, experiments_df)
        with tab3:
            tab_index[3]["func"](experimentset, experiments_df)
        with tab4:
            tab_index[4]["func"](experimentset)

    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("Experiments")
        with col2:
            if st.button("🔄 Refresh", key="refresh_main"):
                st.rerun()

        display_experiment_sets(experiment_sets)


main()
