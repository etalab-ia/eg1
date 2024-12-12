import pandas as pd
from datetime import datetime
import streamlit as st
from utils import fetch
from io import StringIO

def get_experiment_data(exp_id):
    response = fetch("get", f"/experiment/{exp_id}", {"with_dataset": "true"})
    if not response:
        return None

    df = pd.read_json(StringIO(response['dataset']['df']))

    if 'answers' in response:
        answers = {answer['num_line']: answer['answer'] for answer in response['answers']}
        df['answer'] = df.index.map(answers)

    if 'results' in response:
        for result in response['results']:
            metric_name = result['metric_name']
            observations = {obs['num_line']: obs['score'] for obs in result['observation_table']}
            df[f'result_{metric_name}'] = df.index.map(observations)

    dataset_name = response.get('dataset', {}).get('name', 'Unknown Dataset')
    model_name = response.get('model', {}).get('name', 'Unknown Model')

    return df, dataset_name, model_name

def display_experiment_set_overview(experimentset):
    st.write(f"## Overview of experiment set: {experimentset['name']}")
    df = pd.DataFrame([
        {
            "Id": exp["id"],
            "Name": exp["name"],
            "Status": exp["experiment_status"],
            "Created Date": exp["created_at"],
            "Num try": exp["num_try"],
            "Num success": exp["num_success"],
        }
        for exp in experimentset["experiments"]
    ])
    df.sort_values(by='Id', ascending=True, inplace=True)

    row_height = 35  
    header_height = 35  
    border_padding = 5  
    dynamic_height = len(df) * row_height + header_height + border_padding

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=dynamic_height,  
        column_config={"Id": st.column_config.TextColumn(width="small")},
    )
    if df['Num success'].sum() != df['Num try'].sum() and (df['Status'] == 'finished').all():
        st.warning("Warning : all experiments are failed.")

    return df

def display_in_progress():
    st.write("## En cours")

def display_experiment_sets(experiment_sets):
    cols = st.columns(3)
    for idx, exp_set in enumerate(experiment_sets):
        when = datetime.fromisoformat(exp_set["created_at"]).strftime("%d %B %Y")
        with cols[idx % 3]:
            with st.container(border=True):
                if st.button(f"{exp_set['name']}", key=f"exp_set_{idx}"):
                    st.session_state["experimentset"] = exp_set
                    st.rerun()
                st.markdown(exp_set.get("readme", "No description available"))
                col1, col2 = st.columns(2)
                col1.caption(f'Experiment: {len(exp_set["experiments"])} ')
                col2.caption(f"Created the {when}")

def display_experiment_details(experiment_ids):
    selected_exp_id = st.selectbox("Select Experiment ID", experiment_ids)
    if selected_exp_id:
        df_with_results, dataset_name, model_name = get_experiment_data(selected_exp_id)
        if df_with_results is not None:
            st.write(f"### Detailed results of the {selected_exp_id} experiment")
            st.write(f"**Dataset:** {dataset_name}")
            st.write(f"**Model:** {model_name}")
            st.dataframe(df_with_results)
        else:
            st.error("Failed to fetch experiment data")

def main():
    if st.session_state.get("experimentset"):
        experimentset = st.session_state["experimentset"]
        if st.button(":arrow_left: Go back", key="go_back"):
            st.session_state["experimentset"] = None
            st.rerun()
        
        tab1, tab2, tab3 = st.tabs(["Set Overview", "Results", "Detail by experiment id"])
        
        with tab1:
            overview_df = display_experiment_set_overview(experimentset)

        # if all experiments are failed
        if overview_df['Num success'].sum() != overview_df['Num try'].sum() and (overview_df['Status'] == 'finished').all():
            with tab2:
                st.warning("Results cannot be displayed as all experiments have failed.")
            with tab3:
                st.warning("Details cannot be displayed as all experiments have failed.")
        
        else:
            with tab2:
                display_in_progress()
            with tab3:
                experiment_ids = overview_df['Id'].tolist()
                display_experiment_details(experiment_ids)



    else:
        st.title("Experiments (Set)")
        experiment_sets = fetch("get", "/experiment_sets")
        if experiment_sets:
            display_experiment_sets(experiment_sets)

main()