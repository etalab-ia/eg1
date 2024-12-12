import streamlit as st
from utils import fetch
from itertools import groupby
from operator import itemgetter

from streamlit import session_state

session_state.layout = "wide"

def main():
    st.title("Metrics")

    metrics = fetch("get", "/metrics")
    if not metrics:
        return

    main_content, right_menu = st.columns([8, 2])

    # Group metrics by type
    sorted_metrics = sorted(metrics, key=itemgetter('type'))
    grouped_metrics = {k: list(v) for k, v in groupby(sorted_metrics, key=itemgetter('type'))}

    # Main content
    with main_content:
        with st.container():
            st.write("""A metric is a quantitative measure applied to an observation of a model's output.
                     Each metric has a specified "require" constraint that indicates which column in the dataset is needed to perform the measurement.
                     The metric returns a score (float) and may optionally include an observation, which is the intermediate result (such as a judge string output).
                    """)

        for metric_type, metrics_group in grouped_metrics.items():
            # Add an anchor for each metric type
            st.markdown(f"<div id='{metric_type.lower()}'></div>", unsafe_allow_html=True)
            st.header(f"{metric_type.capitalize()} Metrics")
            for metric in metrics_group:
                with st.container():
                    st.markdown(f"<div id='{metric['name'].lower().replace(' ', '-')}'></div>", unsafe_allow_html=True)
                    st.subheader(metric["name"])
                    st.write(f"Required fields: {', '.join(map(lambda x: '**' + x + '**', metric['require']))}")
                    st.write(metric["description"])
                    st.divider()

    # Navigation menu
    with right_menu:
        st.markdown("###### Quick Navigation")
        for metric_type in grouped_metrics.keys():
            metric_id = metric_type.lower()

            # Create a button for each metric type
            st.markdown(f"""
                <a href="#{metric_type}" style="color:grey;"
                   onclick="document.getElementById('{metric_id}').scrollIntoView({{behavior: 'smooth'}});">
                    {metric_type} metrics
                </a><br>
            """, unsafe_allow_html=True)
main()
