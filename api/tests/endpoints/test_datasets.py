import pytest
from fastapi.testclient import TestClient
from api.tests.utils import datasets 
from api.tests.test_api import TestApi
import pandas as pd

class TestEndpointsDataset(TestApi):
    @pytest.mark.parametrize("test_case", [
        {
            "name": "without_output",
            "df_data": {
                'query': [1, 2],
                'colonne2': ['A', 'B']
            }
        },
        {
            "name": "with_output",
            "df_data": {
                'query': [1, 2],
                'output': ['A', 'B']
            }
        }
    ])
    def test_create_dataset(self, client: TestClient, test_case):
        test_df = pd.DataFrame(test_case["df_data"])
        
        response = datasets.create_dataset(
            client=client,
            name_dataset=f"test_dataset_{test_case['name']}",
            df=test_df,
            readme="dataset de test",
            default_metric="test_metric"
        )
        
        assert response.status_code == 200

    def test_read_dataset(self, client: TestClient):
        response = datasets.read_dataset(client)
        print("response: ", response)
        assert response.status_code == 200

    def test_read_dataset_id(self, client: TestClient):
        test_df = pd.DataFrame({
            'query': [1, 2],
            'output': ['A', 'B']
        })
        
        create_response = datasets.create_dataset(
            client=client,
            name_dataset="test_dataset",
            df=test_df,
            readme="dataset de test",
            default_metric="test_metric"
        )
        
        dataset_id = create_response.json()["id"]
        
        response_without_df = datasets.read_dataset_id(client, dataset_id, with_df=False)
        assert response_without_df.status_code == 200
        
        response_with_df = datasets.read_dataset_id(client, dataset_id, with_df=True)
        assert response_with_df.status_code == 200

    def test_read_dataset_id_not_found(self, client: TestClient):
        nonexistent_id = 99999
        response = datasets.read_dataset_id(client, nonexistent_id)
        assert response.status_code == 404