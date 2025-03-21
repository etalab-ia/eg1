from itertools import groupby
from operator import itemgetter
from pathlib import Path
import json
import re
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from utils import fetch, calculate_tokens_per_second

DEFAULT_METRIC = "judge_exactness"

#TODO:
# recup list mtetric du produit et n'afficher que les résultats dessus 


@st.cache_data(ttl=300)
def load_product_config() -> dict:
    """Loads product configuration from YAML file, caching the result."""
    config_path = Path("config") / "products" / "product_config.yml"
    if not config_path.exists():
        st.error(f"Configuration file not found at: {config_path}")
        return {"products": {}}

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return {"products": {}}


def fetch_experiment_results(exp_id: int) -> dict:
    """Fetches results for a single experiment."""
    return fetch("get", f"/experiment/{exp_id}", {"with_dataset": "true"})


@st.cache_data(ttl=300)
def fetch_leaderboard(metric_name: str = DEFAULT_METRIC, dataset_name: str | None = None, limit: int = 10) -> dict:
    """Fetches leaderboard data with caching."""
    endpoint = "/leaderboard"
    params = {"metric_name": metric_name}
    if dataset_name:
        params["dataset_name"] = dataset_name
    return fetch("get", endpoint, params)


@st.cache_data(ttl=300)
def fetch_datasets() -> list[dict]:
    """Fetches datasets with caching."""
    return fetch("get", "/datasets")


def format_column_name(name: str) -> str:
    """Formats column names for better readability."""
    return name.replace("_", " ").title()


def display_model_production(model_info: dict, default_metric: str) -> None:
    """Displays production model information and metrics."""
    st.write("#### Production Model")
    current_model = model_info.get("current", {})
    if not current_model:
        st.info("No production model currently set.")
        return

    model_name = current_model.get("name", "N/A")
    model_id = current_model.get("id", "N/A")
    model_version = current_model.get("version", "N/A")
    model_last_updated = current_model.get("last_updated", "N/A")

    st.write(
        f"Current Model: {model_name} (ID: {model_id}) - "
        f"v{model_version} - Last updated: {model_last_updated}"
    )

    exp_id = current_model.get("id")
    if not exp_id:
        st.warning("Experiment ID not found for the current model.")
        return

    try:
        results = fetch_experiment_results(exp_id)
        if not results or "results" not in results:
            st.info(f"No results found for experiment {exp_id}")
            return

        metrics_data = {"Model": model_name}
        tokens, times = [], []

        for result in results["results"]:
            metric_name = result["metric_name"]
            scores = [obs["score"] for obs in result["observation_table"] if "score" in obs]
            if not scores:
                st.warning(f"No scores found for metric {metric_name} in experiment {exp_id}")
                continue

            mean_score = np.mean(scores)

            if metric_name == "nb_tokens_completion":
                tokens = scores
            elif metric_name == "generation_time":
                times = scores
            else:
                column_name = f"{format_column_name(metric_name)} Score" if metric_name == default_metric else metric_name
                metrics_data[column_name] = round(mean_score, 2)

        if tokens and times:
            tokens_per_second = [
                calculate_tokens_per_second(t, tm) for t, tm in zip(tokens, times)
            ]
            tokens_per_second = [tps for tps in tokens_per_second if tps is not None]

            if tokens_per_second:
                metrics_data["tokens_per_second"] = round(np.mean(tokens_per_second), 2)

        df = pd.DataFrame([metrics_data])
        st.dataframe(df)

    except Exception as e:
        st.error(f"Error fetching results for experiment {exp_id}: {str(e)}")


def _all_equal(lst: list) -> bool:
    """Checks if all elements in a list are equal."""
    return all(x == lst[0] for x in lst)


def _remove_commons_items(model_params: List[Dict], first: bool = True) -> List[Dict]:
    """
    Removes common items from a list of dictionaries.
    Handles nested dictionaries recursively.
    """
    if first:
        model_params = deepcopy(model_params)

    common_keys = set.intersection(*(set(d.keys()) for d in model_params))
    for k in common_keys:
        values = [d.get(k) for d in model_params]

        if _all_equal(values):
            for d in model_params:
                d.pop(k, None)
        elif all(isinstance(d.get(k), dict) for d in model_params):
            dicts = [d.get(k) for d in model_params]
            modified_dicts = _remove_commons_items(
                [d for d in dicts if isinstance(d, dict)], first=False
            )

            idx = 0
            for i, d in enumerate(model_params):
                if isinstance(d.get(k), dict):
                    if idx < len(modified_dicts):
                        d[k] = modified_dicts[idx]
                        idx += 1
        elif all(isinstance(d.get(k), list) for d in model_params):
            pass  # Handle lists if needed

    return model_params


def _rename_model_variants(experiments: List[Dict]) -> None:
    """
    Inplace adds a _name attribute to experiment if several model names are equal,
    to help distinguish them.
    """
    names = [exp["model"]["name"] for exp in experiments if exp.get("model")]
    if len(set(names)) == len(names):
        return

    names_data = []
    for i, exp in enumerate(experiments):
        if not exp.get("model"):
            continue

        name = exp["model"]["name"]
        _name = name
        suffix = ""
        match = re.search(r"__\d+$", name)
        if match:
            _name, suffix = name.rsplit("__", 1)

        names_data.append(
            {
                "pos": i,
                "name": name,
                "_name": _name,
                "suffix": suffix,
            }
        )

    model_names = defaultdict(list)
    for item in names_data:
        model_names[item["_name"]].append(item["pos"])

    for _name, ids in model_names.items():
        if len(ids) <= 1:
            continue

        model_params = [
            (experiments[i]["model"].get("sampling_params") or {})
            | (experiments[i]["model"].get("extra_params") or {})
            for i in ids
        ]

        model_diff_params = _remove_commons_items(model_params)

        for model in names_data:
            pos = next((x for x in ids if model["pos"] == x), None)
            if pos is None:
                continue

            variant = model_diff_params[ids.index(pos)]
            if variant:
                variant = json.dumps(variant, sort_keys=True)
                variant = variant.replace('"', "").replace(" ", "")
                experiments[pos]["_model"] = "#".join([_name, variant]) + model["suffix"]


def add_repeat_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'repeat' column to the DataFrame based on experiment name pattern."""

    def check_repeat(name: str) -> bool:
        if pd.isna(name):
            return False
        return bool(re.search(r"__\d+$", str(name)))

    df['repeat'] = df['Experiment name'].apply(check_repeat)
    return df


def process_leaderboard_data(
    leaderboard_data: Dict,
    metric_name: str,
    metrics_list_for_decision: List[str],
    current_model_id: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Processes leaderboard data to create a Pandas DataFrame.

    Args:
        leaderboard_data (Dict): Raw leaderboard data.
        metric_name (str): The primary metric to display.
        metrics_list_for_decision (List[str]): List of metrics for decision-making.
        current_model_id (Optional[int]): ID of the current production model.

    Returns:
        Optional[pd.DataFrame]: Processed leaderboard DataFrame, or None if no data.
    """
    experiments = []
    exp_mapping = {}

    for entry in leaderboard_data.get("entries", []):
        if current_model_id and entry["experiment_id"] == current_model_id:
            continue

        exp = {
            "model": {
                "name": entry["model_name"],
                "sampling_params": entry.get("sampling_param", {}),
                "extra_params": entry.get("extra_param", {}),
            }
        }
        experiments.append(exp)
        exp_mapping[entry["experiment_id"]] = exp

    _rename_model_variants(experiments)

    entries = []
    for entry in leaderboard_data.get("entries", []):
        if current_model_id and entry["experiment_id"] == current_model_id:
            continue

        exp = exp_mapping.get(entry["experiment_id"], {})
        model_name = entry["model_name"]
        renamed_model = exp.get("_model", model_name)

        params = {**entry.get("sampling_param", {}), **entry.get("extra_param", {})}
        parameters = json.dumps(params, sort_keys=True, ensure_ascii=False) if params else ""

        processed_entry = {
            "Experiment ID": entry["experiment_id"],
            "Experiment name": entry["experiment_name"],
            "Model": model_name,
            "Model_renamed": renamed_model,
            "Parameters": parameters,
            "Created at": entry["created_at"],
            "experiment_set_id": entry["experiment_set_id"],
            "experiment_set_name": entry["experiment_set_name"],
            f"{format_column_name(metric_name)} Score": entry["main_metric_score"],
        }

        processed_entry.update(
            {format_column_name(metric): score for metric, score in entry.get("other_metrics", {}).items()}
        )

        tokens_per_second = calculate_tokens_per_second(
            entry["other_metrics"].get("nb_tokens_completion"),
            entry["other_metrics"].get("generation_time"),
        )
        processed_entry["tokens_per_second"] = tokens_per_second

        entries.append(processed_entry)

    if not entries:
        return None

    df = pd.DataFrame(entries)
    df = add_repeat_column(df)

    score_column = f"{format_column_name(metric_name)} Score"

    # Separate repeat and non-repeat experiments
    df_repeat_false = df[df["repeat"] == False]
    df_repeat_false[f"{score_column}_mean"] = df_repeat_false[score_column]
    df_repeat_false['count'] = 1

    df_repeat_true = df[df["repeat"] == True]

    # Aggregate repeated experiments
    if not df_repeat_true.empty:
        first_columns = [
            "Model",
            "Model_renamed",
            "Parameters",
            "Created at",
            "experiment_set_id",
            "experiment_set_name",
        ]
        numeric_columns = df_repeat_true.select_dtypes(include=[np.number]).columns

        grouped = df_repeat_true.groupby("Model_renamed").agg(
            {
                **{col: "first" for col in first_columns},
                **{
                    col: ["mean", "std"] for col in numeric_columns if col not in first_columns
                }, 
            }
        )
        grouped['count'] = df_repeat_true.groupby("Model_renamed").size()

        result_repeat_true = pd.DataFrame()

        for col in first_columns:
            result_repeat_true[col] = grouped[col]

        result_repeat_true['count'] = grouped['count']  

        for column in numeric_columns:
            if column not in first_columns:
                mean = grouped[(column, "mean")].round(2)
                std = grouped[(column, "std")].round(2)
                result_repeat_true[f"{column}_mean"] = mean
                result_repeat_true[column] = mean.astype(str) + " ± " + std.astype(str)

        result_repeat_true.reset_index(drop=True, inplace=True)
    else:
        result_repeat_true = pd.DataFrame()

    # Concatenate results 
    final_df = pd.concat([df_repeat_false, result_repeat_true], ignore_index=True)

    # Sort by metric
    sort_column = f"{score_column}_mean" if f"{score_column}_mean" in final_df.columns else score_column

    final_df.sort_values(sort_column, ascending=False, na_position="last", inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    final_df["Rank"] = final_df.index + 1

    # Drop unnecessary columns
    columns_to_drop = (
        [col for col in final_df.columns if col.endswith("_mean")]
        + ["Experiment ID", "Experiment name", "Model_renamed", "experiment_set_name", "repeat"]
    )
    final_df.drop(columns=columns_to_drop, inplace=True, errors='ignore') # added errors='ignore'

    # Define column order
    fixed_columns = ["Rank", "Model", "Parameters", "Created at", score_column]
    col_decision = [col for col in final_df.columns if col in metrics_list_for_decision]
    other_columns = [col for col in final_df.columns if col not in fixed_columns + col_decision]
    column_order = fixed_columns + col_decision + other_columns

    return final_df[[col for col in column_order if col in final_df.columns]]


def display_dataset_and_metrics(product_info: dict, datasets: list[dict]) -> str:
    """Displays dataset information and product metrics side by side."""
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Dataset")
        dataset_info = product_info.get("dataset", {})
        st.write(f"**Name:** {dataset_info.get('name', 'N/A')}")

        product_dataset_name = dataset_info.get("name")
        default_metric = DEFAULT_METRIC

        if product_dataset_name and datasets:
            default_metric = next(
                (
                    dataset.get("default_metric", DEFAULT_METRIC)
                    for dataset in datasets
                    if dataset["name"] == product_dataset_name
                ),
                DEFAULT_METRIC,
            )

        st.write(f"**Default metric:** {default_metric}")

    with col2:
        st.write("#### Necessary metrics for decision")
        if "metrics" not in product_info or not product_info["metrics"]:
            st.info("No metrics available for this product")
        else:
            metrics_list = [metric.replace('_', ' ').capitalize() for metric in product_info["metrics"]]
            st.write(", ".join(metrics_list))

    return default_metric


def get_base_url() -> str:
    """Constructs the base URL for the Streamlit app."""
    base_url = st.get_option('server.baseUrlPath')
    return base_url if base_url else "/"


def create_experiment_set_link(experiment_set_id: Any, experiments_url: str) -> Optional[str]:
    """Creates a hyperlink to an experiment set if a valid ID is provided."""
    if pd.isna(experiment_set_id) or not experiment_set_id:
        return None
    try:
        return f"{experiments_url}{int(experiment_set_id)}"
    except (ValueError, TypeError):
        return None


def main() -> None:
    """Main function to run the Streamlit app."""
    st.title("Products Metrics Leaderboard")

    product_config = load_product_config()

    if not product_config.get("products"):
        st.warning("No products configured yet.")
        return

    product_tabs = st.tabs(
        [product_info["name"] for product_info in product_config["products"].values()]
    )

    for tab, product_info in zip(product_tabs, product_config["products"].values()):
        with tab:
            st.header(f"{product_info['name'].replace('_', ' ')}")

            datasets = fetch_datasets()
            default_metric = display_dataset_and_metrics(product_info, datasets)

            st.divider()

            # model in production
            if "production_model" in product_info:
                display_model_production(product_info["production_model"], default_metric)
            else:
                st.warning("No production model information available")

            # models tests in exp
            current_model_id = (
                product_info.get("production_model", {})
                .get("current", {})
                .get("id")
            )
            product_dataset_name = product_info.get("dataset", {}).get("name")
            metrics_list_for_decision = [metric for metric in product_info["metrics"]]

            leaderboard_data = fetch_leaderboard(default_metric, product_dataset_name)
            df_leaderboard = process_leaderboard_data(
                leaderboard_data, default_metric, metrics_list_for_decision, current_model_id
            )

            if df_leaderboard is not None:
                base_url = get_base_url()
                experiments_url = f"{base_url}experiments_set?expset="

                df_leaderboard["Experiment Set Link"] = df_leaderboard["experiment_set_id"].apply(
                    lambda x: create_experiment_set_link(x, experiments_url)
                )
                df_leaderboard.drop(columns="experiment_set_id", inplace=True)

                st.write("#### Leaderboard")
                st.data_editor(
                    df_leaderboard,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Experiment Set Link": st.column_config.LinkColumn(
                            "Experiment Set Link"
                        )
                    },
                )
            else:
                st.write("No data available for the leaderboard.")


main()
