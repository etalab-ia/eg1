import streamlit as st
from utils import fetch

from streamlit import session_state

session_state.layout = 'wide'

def main():
    st.title("Datasets")

    datasets = fetch("get", "/datasets")
    if not datasets:
        return


    main_content, right_menu = st.columns([8, 2])

    # Main content
    with main_content:
        with st.container():
            st.write("""Avalaible datasets
                     """)

        for dataset in datasets:
            with st.container():
                # Add an anchor for navigation
                st.markdown(f"<div id='{dataset['name'].lower().replace(' ', '-')}'></div>", unsafe_allow_html=True)
                st.subheader(dataset['name'])
                st.write(f"readme: {dataset['readme']}")
                st.write(f"size: {dataset['size']}")
                st.divider()

    # Navigation menu
    with right_menu:
        st.markdown("###### Quick Navigation")
        for dataset in datasets:
            dataset_id = dataset['name'].lower().replace(' ', '-')
            st.markdown(f"""
                <a href="#{dataset_id}" style="color:grey;"
                   onclick="document.getElementById('{dataset_id}').scrollIntoView({{behavior: 'smooth'}});">
                    {dataset['name']}
                </a><br>
            """, unsafe_allow_html=True)


main()
