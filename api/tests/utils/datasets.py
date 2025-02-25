from fastapi.testclient import TestClient
from api.config import API_BASE_URL


ROOT_PATH = API_BASE_URL


def read_dataset(client: TestClient):
    return client.get(f"{ROOT_PATH}/v1/datasets")

def read_dataset_id(client: TestClient, id: int, with_df: bool = False):
    params = {"with_df": with_df}
    return client.get(f"{ROOT_PATH}/v1/dataset/{id}", params=params)

def create_dataset(client: TestClient, name_dataset, df, readme, default_metric):
    data = {
        "name": name_dataset,
        "df": df.to_json(),
        "readme": readme,
        "default_metric": default_metric
    }
    
    return client.post(f"{ROOT_PATH}/v1/dataset", json=data)

