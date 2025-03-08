import streamlit as st

ROUTES = [
    {
        "id": "home",
        "path": "views/home.py",
        "title": "Home",
        "icon": ":material/home:",
    },
    # {
    #     "id": "leaderboard",
    #     "path": "views/leaderboard.py",
    #     "title": "Leaderboard",
    #     "description": "Best models ranking list",
    #     "icon": ":material/trophy:",
    # },
        {
        "id": "leaderboard",
        "path": "views/product_leaderboard.py",
        "title": "Leaderboard",
        "description": "Best models ranking list (by products)",
        "icon": ":material/trophy:",
    },
    {
        "id": "experiments_set",
        "path": "views/experiments_set.py",
        "title": "Experiments Set",
        "description": "Navigate the experiments collections groups",
        "icon": ":material/experiment:",
    },
    {
        "id": "experiments",
        "path": "views/experiments.py",
        "title": "Experiment",
        "description": "Navigate the experiments not attached to an experiment group",
        "icon": ":material/dashboard:",
    },
    {
        "id": "datasets",
        "path": "views/datasets.py",
        "title": "Datasets",
        "description": "A list of available datasets",
        "icon": ":material/database:",
    },
    {
        "id": "metrics",
        "path": "views/metrics.py",
        "title": "Metrics",
        "description": "A list of available metrics",
        "icon": ":material/visibility:",
    },
    {
        "id": "ops",
        "path": "views/ops.py",
        "title": "Ops",
        "description": "Ops analysis",
        "icon": ":material/settings:",
    },
]


def get_page(route: str | dict):
    if isinstance(route, str):
        route = next((x for x in ROUTES if x["id"] == route), None)
    elif isinstance(route, dict):
        pass
    else:
        raise ValueError("Unsupported data type for route: %s" % type(route))

    if route is None:
        raise ValueError("Route not found: %s" % route)

    page = st.Page(
        route["path"], title=route["title"], icon=route.get("icon"), url_path=route["id"]
    )
    return page
