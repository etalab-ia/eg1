import importlib.metadata
import logging
import os
import tempfile

import dotenv

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

dotenv.load_dotenv()

#######################################################################
### App metadata
#######################################################################

APP_NAME = "eg1" # redundant wuth pyproject.name
APP_DESCRIPTION = "Albert Evaluations API"
CONTACT = {
    "name": "Etalab - Datalab",
    "url": "https://www.etalab.gouv.fr/",
    "email": "etalab@modernisation.gouv.fr",
}

try:
    APP_VERSION = importlib.metadata.version(APP_NAME)
except importlib.metadata.PackageNotFoundError:
    logging.warning(f"Package {APP_NAME} is not installed.")
    APP_VERSION = "0.0"


#######################################################################
### Commons
#######################################################################

ENV = os.getenv("ENV", "dev")
assert ENV in ["unittest", "dev", "prod"], "wrong ENV value"
BACKEND_CORS_ORIGINS = ["*"]
ALBERT_API_URL = os.getenv("ALBERT_API_URL")
ALBERT_API_KEY = os.getenv("ALBERT_API_KEY")
API_PREFIX = ""

# Runners
MAX_CONCURRENT_TASKS = 4 # 8 ok, 16 hard !

#######################################################################
### Environment specific
#######################################################################

if ENV == "unittest":
    API_BASE_URL = "http://localhost:8000" + API_PREFIX
    DB_NAME = "eg1-unittest"
    DATABASE_URI = "sqlite:///" + os.path.join(tempfile.gettempdir(), f"{DB_NAME}-sqlite3.db")
elif ENV == "dev":
    API_BASE_URL = "http://localhost:8000" + API_PREFIX
    DB_NAME = "eg1_dev"
    DATABASE_URI = os.getenv(
        "POSTGRES_URI", "postgresql+psycopg2://postgres:changeme@localhost:5432/eg1_dev"
    )
else:
    API_BASE_URL = "http://localhost:8000" + API_PREFIX
    DB_NAME = "eg1"
    DATABASE_URI = os.getenv(
        "POSTGRES_URI", "postgresql+psycopg2://postgres:changeme@localhost:5432/eg1"
    )
