import requests
import streamlit as st
from typing import Optional
import hashlib

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


def hash_string(input_string, bits=8):
    hash_object = hashlib.sha256(input_string.encode())
    hash_hex = hash_object.hexdigest()
    return hash_hex[:bits]


def calculate_tokens_per_second(tokens: Optional[int], time: Optional[float]) -> Optional[float]:
    return (
        round(tokens / time, 1) if tokens is not None and time is not None and time != 0 else None
    )
