import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000/v1"


def fetch(method, endpoint, data=None):
    func = getattr(requests, method)
    q = ""
    kw = {}
    if method == "get" and data:
        q = "?" + "&".join([f"{k}={v}" for k, v in data.items()])
    elif data:
        kw["json"] = data

        response = func(f"{API_BASE_URL}{endpoint}{q}", **kw)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch data from {endpoint}.")
        return None
