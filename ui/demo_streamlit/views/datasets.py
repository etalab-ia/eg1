from datetime import datetime
import streamlit as st
from utils import fetch

from streamlit import session_state

session_state.layout = "wide"


def main():
    st.title("Datasets")

    datasets = fetch("get", "/datasets")
    if not datasets:
        return

    main_content, right_menu = st.columns([8, 2])

    with main_content:
        with st.container():
            st.write("""Avalaible datasets
                     """)

        for dataset in datasets:
            when = datetime.fromisoformat(dataset["created_at"]).strftime("%d %B %Y")
            with st.container():
                st.markdown(
                    f"<div id='{dataset['name'].lower().replace(' ', '-')}'></div>",
                    unsafe_allow_html=True,
                )
                st.subheader(dataset["name"])
                st.write(
                    f"Columns: {', '.join(map(lambda x: '**' + x + '**', dataset["columns"]))}"
                )
                st.markdown(dataset.get("readme", "No description available"))
                col1, col2, col3 = st.columns([1 / 6, 2 / 6, 3 / 6])
                with col1:
                    st.caption(f'id: {dataset["id"]} ')
                with col2:
                    st.caption(f"Rows: {dataset['size']}")
                with col3:
                    st.caption(f"Created the {when}")
                st.divider()

    with right_menu:
        st.markdown("###### Quick Navigation")
        for dataset in datasets:
            dataset_id = dataset["name"].lower().replace(" ", "-")
            st.markdown(
                f"""
                <a href="#{dataset_id}" style="color:grey;"
                   onclick="document.getElementById('{dataset_id}').scrollIntoView({{behavior: 'smooth'}});">
                    {dataset['name']}
                </a><br>
            """,
                unsafe_allow_html=True,
            )


main()
