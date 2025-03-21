import json
import re
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import fetch


@st.cache_data
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


@st.cache_data
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
                    st.caption(f"id: {exp_set['id']} ")
                with col2:
                    st.caption(f"Experiments: {len(exp_set['experiments'])} ")
                with col3:
                    st.caption(f"Created on {when}")


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


def _format_experiments_score_df(experiments: list, df: pd.DataFrame) -> (bool, pd.DataFrame):
    experiment_ids = [exp["id"] for exp in experiments]
    experiment_names = [exp["name"] for exp in experiments]
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
        #df["name"] = experiment_names
        df = df[["Id", "model"] + [col for col in df.columns if col not in ["Id", "model"]]]
        has_repeat = False
    else:
        df = result

    # @DEBUG: when +- is used, the sorting does not work.
    #default_sort_metric = _find_default_sort_metric(df.columns)
    #if default_sort_metric in df.columns:
    #    df = df.sort_values(by=f"{default_sort_metric}", ascending=False)
    # @DEBUG: Id does not exist for "repeat" case
    #df = df.sort_values(by="Id", ascending=True)

    return has_repeat, df


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
            row["model"] = exp.get("_model") or exp["model"]["aliased_name"] or exp["model"]["name"]
            row_support["model"] = (
                exp.get("_model") or exp["model"]["aliased_name"] or exp["model"]["name"]
            )

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
    try:
        has_repeat, df = _format_experiments_score_df(experiments, df)
    except ValueError as err:
        st.error("No result found yet, please try again later")
        raise err
        return

    df_support = pd.DataFrame(rows_support)
    df_support = _sort_columns(df_support, [])
    _, df_support = _format_experiments_score_df(experiments, df_support)

    st.write("**Score:** Averaged score on experiments metrics")
    if has_repeat:
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

        report_data.append({
            "Experiment Set Name": exp_set["name"],
            "Status": status["text"],
            "Total Experiments": len(exp_set["experiments"]),
            "Answer Tries": counts["total_answer_tries"],
            "Answer Successes": counts["total_answer_successes"],
            "Observation Tries": counts["total_observation_tries"],
            "Observation Successes": counts["total_observation_successes"],
            "Has Failure": has_failure,
        })

    report_df = pd.DataFrame(report_data)
    st.dataframe(
        report_df,
        use_container_width=True,
        hide_index=True,
    )


def process_experiment(exp):
    exp_id = exp["id"]
    experiment = fetch("get", f"/experiment/{exp_id}", {"with_dataset": "true"})
    return experiment


def update_model_data(model_data, experiment):
    model_name = experiment.get("model", {}).get("name") or experiment.get("model", {}).get("aliased_name", "Unknown Model")
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
    total = row['failed'] + row['no_failed']
    return row['failed'] / total if total > 0 else 0


def report_model_and_metric(experimentset):
    """Analyzes experiment statuses by model and metric, including failed experiments and failure rates."""
    model_data = defaultdict(lambda: {"finished": 0, "running":0, "running_answers": 0, "running_metrics": 0, "pending": 0, "failed": 0, "no_failed": 0})
    metric_data = defaultdict(lambda: {"finished": 0, "running": 0, "pending": 0, "failed": 0, "no_failed": 0})

    for exp in experimentset["experiments"]:
        experiment = process_experiment(exp)
        if experiment:
            update_model_data(model_data, experiment)
            update_metric_data(metric_data, experiment)

    model_report = pd.DataFrame.from_dict(model_data, orient="index")
    model_report["Total"] = model_report[["finished", "running", "pending"]].sum(axis=1)
    model_report["Failure Rate"] = model_report.apply(calculate_failure_rate, axis=1)

    columns_order = ["finished", "running", "pending", "Total", "failed", "no_failed", "Failure Rate"]
    model_report = model_report[columns_order]
    model_for_graph = model_report.copy()
    model_report.columns = [col.title().replace("_", " ") for col in model_report.columns]

    model_report.columns = pd.MultiIndex.from_tuples([
        ("Status", "Finished"), ("Status", "Running"), ("Status", "Pending"), ("Status", "Total"),
        ("Failure Analysis", "Failed"), ("Failure Analysis", "No Failed"), ("Failure Analysis", "Failure Rate")
    ])

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

    metric_report.columns = pd.MultiIndex.from_tuples([
        ("Status", "Finished"), ("Status", "Running"), ("Status", "Pending"), ("Status", "Total"),
        ("Failure Analysis", "Failed"), ("Failure Analysis", "No Failed"), ("Failure Analysis", "Failure Rate")
    ])

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

    col_showhead1, col_showhead2 = st.columns([1, 10])
    with col_showhead1:
        st.markdown(f"**Id**: {experimentset['id']}")
    with col_showhead2:
        st.markdown(f"**Readme:** {experimentset.get('readme', 'No description available')}")

    finished_ratio = int(
        counts["total_observation_successes"] / counts["observation_length"] * 100
    )
    failure_ratio = int(
        (counts["total_observation_tries"] - counts["total_observation_successes"])
        / counts["observation_length"]
        * 100
    )

    metric_status = f"**Metric status:** Finished: {finished_ratio}%"
    if failure_ratio > 0:
        metric_status += f" &nbsp;&nbsp;&nbsp; Failure: <span style='color:red;'>{failure_ratio}%</span>"

    st.markdown(metric_status, unsafe_allow_html=True)

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

        col_head1, col_head2 = st.columns([1, 8])

        with col_head1:
            if st.button("← Go Back"):
                st.session_state["experimentset"] = None
                st.query_params.pop("expset")
                st.rerun()

        with col_head2:
            show_header(st.session_state['experimentset'])

        tab1, tab2, tab3, tab4 = st.tabs([
            "⭐ Scores", 
            "📊 Set Overview", 
            "📝 Details by Experiment Id", 
            "🚨 Ops Analysis"
        ])

        experimentset = st.session_state["experimentset"]
        experiments_df = pd.DataFrame([
            {     "Id": exp["id"],
                    "Name": exp["name"],
                    "Status": exp["experiment_status"],
                    "Created at": exp["created_at"],
                    "Num try": exp["num_try"],
                    "Num success": exp["num_success"],
                    "Num observation try": exp["num_observation_try"],
                    "Num observation success": exp["num_observation_success"],
 
             }
            for exp in experimentset["experiments"]
        ])

        with tab1:
            col_score1, col_score2 = st.columns([4, 1])
            with col_score1:
                st.subheader("Scores")
            with col_score2:
                if st.button("🔄 Refresh Scores", key="refresh_scores"):
                    st.cache_data.clear()
            display_experiment_set_score(experimentset, experiments_df)

        with tab2:
            col_overview1, col_overview2 = st.columns([4, 1])
            with col_overview1:
                st.subheader("Overview")
            with col_overview2:
                if st.button("🔄 Refresh Overview", key="refresh_overview"):
                    st.cache_data.clear()
            display_experiment_set_overview(experimentset, experiments_df)
            
        with tab3:
            col_detail1, col_detail2 = st.columns([4, 1])
            with col_detail1:
                st.subheader("Details")
            with col_detail2:
                if st.button("🔄 Refresh Details", key="refresh_details"):
                    st.cache_data.clear()
            display_experiment_details(experimentset, experiments_df)


        with tab4:
            col_report1, col_report2 = st.columns([4, 1])
            with col_report1:
                st.subheader("Ops analysis")
            with col_report2:
                if st.button("🔄 Refresh Report", key="refresh_report"):
                    st.cache_data.clear()
            display_ops_analysis(experimentset)

    else:
        st.title("Experiment (Sets)")
        col_main1, col_main2 = st.columns([4, 1])
        with col_main2:
            if st.button("🔄 Refresh List", key="refresh_main"):
                st.cache_data.clear()
        display_experiment_sets(experiment_sets)

main()
