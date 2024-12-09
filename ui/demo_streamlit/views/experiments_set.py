import pandas as pd
from datetime import datetime
import streamlit as st
from utils import fetch
from routes import get_page


def display_adaptive_dataframe(df):
    row_height = 35  
    padding = 40 
    num_rows = len(df)
    height = min(num_rows * row_height + padding, 600)  

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=height,
        column_config={
            "Id": st.column_config.TextColumn(width="small"),
        },
    )

def main():
    # Display selected experiment set overview
    if st.session_state.get("experimentset"):
        experimentset = st.session_state["experimentset"]
        if st.button(f":material/arrow_back: Go back"):
            st.session_state["experimentset"] = None
            st.rerun()
        st.write(f"## Overview of experiment set: {experimentset['name']}")

        # Convert experiments list to a pandas DataFrame
        df = pd.DataFrame(
            [
                {
                    "Id": exp["id"],
                    "Name": exp["name"],
                    "Status": exp["experiment_status"],
                    "Created Date": exp["created_at"],
                    "Num try": exp["num_try"],
                    "Num success": exp["num_success"],
                }
                for exp in experimentset["experiments"]
            ]
        )
        df.sort_values(by='Id', ascending=True, inplace=True)

        # Show the table, adaptative function number of row in dataframe
        display_adaptive_dataframe(df)

        return

    st.title("Experiments (Set)")

    # Fetch experiment sets
    experiment_sets = fetch("get", "/experiment_sets")
    if not experiment_sets:
        return

    # Create a grid of cards (3 columns)
    cols = st.columns(3)

    for idx, exp_set in enumerate(experiment_sets):
        when = datetime.fromisoformat(exp_set["created_at"]).strftime("%d %B %Y")

        with cols[idx % 3]:
            # Create a card-like container
            with st.container(border=True):
                # st.page_link(get_page("experiment"), label=f"**{exp_set['name']}**") # Issue with https://github.com/streamlit/streamlit/issues/9195
                if st.button(f"{exp_set['name']}"):
                    st.session_state["experimentset"] = exp_set
                    st.rerun()
                st.markdown(exp_set.get("readme", "No description available"))
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f'Experiment: {len(exp_set["experiments"])} ')
                with col2:
                    st.caption(f"Created the {when}")


main()
