import os
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from api.db import get_db
from api.main import app
from api.tests.db.init_db import init_db as test_init_db
from api.tests.db.base import Base
from api.tests.db.session import engine

os.environ["ENV"] = "unittest"
# from db.create_admin_user import get_or_create_admin_user
# from db.session import SessionLocal

APP_FOLDER = Path(__file__).parents[2]

def override_get_db():
    try:
        db = test_init_db()
        yield db
    finally:
        db.close()

@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="module")
def client() -> Generator:
    # Override the database dependency
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


#
# DB
#


# @pytest.fixture(scope="session")
# def db() -> Generator:
#     print("Setup session...")
#     try:
#         session = SessionLocal()
#         yield session
#     finally:
#         session.close()
#     print("Teardown session.")
