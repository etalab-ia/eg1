from datetime import datetime
from enum import Enum
from io import StringIO
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, create_model
from sqlalchemy.orm import Session

import api.models as models
from api.errors import SchemaError
from api.metrics import metric_registry
from api.utils import build_param_grid


#
# Custom BaseModel
#
class EgBaseModel(BaseModel):
    def recurse_table_init(self, db: Session) -> dict:
        obj = self.model_dump()
        for k, v in obj.items():
            sub_schema = getattr(self, k)
            if hasattr(sub_schema, "to_table_init"):
                obj[k] = sub_schema.to_table_init(db)
            elif isinstance(sub_schema, list):
                obj[k] = [
                    o.recurse_table_init(db) if isinstance(o, BaseModel) else o for o in sub_schema
                ]

        return obj

    def to_table_init(self, db: Session):
        return self.model_dump()


#
# Enum
#

MetricEnum = Enum(
    "MetricEnum", {name: name for name in metric_registry.get_metric_names()}, type=str
)


class ExperimentStatus(str, Enum):
    pending = "pending"
    running_answers = "running_answers"
    running_metrics = "running_metrics"
    finished = "finished"


class MetricStatus(str, Enum):
    pending = "pending"
    running = "running"
    finished = "finished"


#
# Dataset
#


class DatasetBase(EgBaseModel):
    name: str

    model_config = ConfigDict(from_attributes=True)


class DatasetCreate(DatasetBase):
    df: str  # from_json

    def to_table_init(self, db: Session) -> dict:
        obj = self.recurse_table_init(db)

        # Handle dataframe
        try:
            df = pd.read_json(StringIO(self.df))
        except ValueError:
            raise SchemaError("'df' should be a readable dataframe. Use df.to_json()...")

        has_query = "query" in df.columns
        has_output = "output" in df.columns
        has_output_true = "output_true" in df.columns

        if not (has_query or has_output):
            raise SchemaError("Your dataset needs at least a column 'query' or 'ouput'.")

        return {
            "has_query": has_query,
            "has_output": has_output,
            "has_output_true": has_output_true,
            "size": len(df),
            **obj,
        }


class Dataset(DatasetBase):
    id: int
    has_query: bool
    has_output: bool
    has_output_true: bool
    size: int


class DatasetFull(Dataset):
    df: str  # from_json


#
# Model
#


class ModelBase(EgBaseModel):
    name: str
    base_url: str
    api_key: str
    prompt_system: str | None = None
    sampling_params: dict | None = None
    extra_params: dict | None = None

    model_config = ConfigDict(from_attributes=True)


class ModelCreate(ModelBase):
    pass


class Model(ModelBase):
    id: int


#
# Result
#


class Answer(EgBaseModel):
    id: int
    #created_at: datetime
    answer: str | None
    num_line: int
    error_msg: str | None
    execution_time: int | None

    model_config = ConfigDict(from_attributes=True)


class Observation(EgBaseModel):
    id: int
    created_at: datetime
    score: float | None
    observation: str | None  # json
    num_line: int
    error_msg: str | None
    execution_time: int | None

    model_config = ConfigDict(from_attributes=True)


class ResultBase(EgBaseModel):
    metric_name: MetricEnum

    model_config = ConfigDict(from_attributes=True)


class ResultCreate(ResultBase):
    experiment_id: int | None = None

    def to_table_init(self, db):
        obj = self.recurse_table_init(db)
        return {"metric_status": "pending", **obj}


class Result(ResultBase):
    id: int
    created_at: datetime
    observation_table: list[Observation]
    num_try: int
    num_success: int
    experiment_id: int
    metric_status: MetricStatus


ResultUpdate = create_model(
    "ResultUpdate",
    **{
        field_name: (Optional[field.annotation], None)
        for field_name, field in Result.__fields__.items()
    },
    __base__=Result,
)

#
# Experiment
#


class ExperimentBase(EgBaseModel):
    name: str
    readme: str | None = None
    experiment_set_id: int | None = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class ExperimentCreate(ExperimentBase):
    metrics: list[MetricEnum]
    dataset: DatasetCreate | str
    model: ModelCreate | None = None

    def to_table_init(self, db: Session) -> dict:
        obj = self.recurse_table_init(db)

        # Handle Dataset
        if isinstance(self.dataset, str):
            dataset = db.query(models.Dataset).filter_by(name=self.dataset).first()
            if not dataset:
                raise SchemaError("Dataset not found")
        else:
            dataset = models.Dataset(**obj["dataset"])
        obj["dataset"] = dataset

        # Handle Results
        results = []
        for metric_name in self.metrics:
            results.append(ResultCreate(metric_name=metric_name).to_table_init(db))
        obj["results"] = results

        # Validate Model and metric compatibility
        mr = metric_registry
        needs_query = any("query" in mr.get_metric(m).require for m in self.metrics)
        needs_output = any("output" in mr.get_metric(m).require for m in self.metrics)
        needs_output_true = any("output_true" in mr.get_metric(m).require for m in self.metrics)
        if needs_query and not dataset.has_query:
            raise SchemaError("You need to provide a query for this metric. ")
        if needs_output and not self.model and not dataset.has_output:
            raise SchemaError(
                "You need to provide an answer for this metric. "
                "Either set a model to generate it or provide a dataset with the 'output' field."
            )
        if needs_output and not dataset.has_output and not dataset.has_query:
            raise SchemaError(
                "You need to provide an answer for this metric. "
                "Either provide a dataset with the 'query' field to generate the answer or with an 'output' field if have generated it yourself."
            )

        if needs_output_true and not dataset.has_output_true:
            raise SchemaError(
                "You need to provide a ground truth for this metric. "
                "Your dataset needs to have an 'output_true' field."
            )

        if dataset.has_output and self.model:
            raise SchemaError(
                "You can't give at the same time a model and a dataset with an answer ('output' column). "
                "Gives either one or the other."
            )

        return {
            "experiment_status": "pending",
            **obj,
        }


class Experiment(ExperimentBase):
    id: int
    created_at: datetime
    experiment_status: ExperimentStatus
    num_try: int
    num_success: int

    dataset: Dataset
    model: Model | None


class ExperimentWithResults(Experiment):
    results: list[Result] | None


class ExperimentWithAnswers(Experiment):
    answers: list[Answer] | None


class ExperimentFull(Experiment):
    answers: list[Answer] | None
    results: list[Result] | None


# For the special `metrics` input
class ExperimentExtra(Experiment, ExperimentCreate):
    pass


ExperimentUpdate = create_model(
    "ExperimentUpdate",
    **{
        field_name: (Optional[field.annotation], None)
        for field_name, field in ExperimentExtra.__fields__.items()
    },
    __base__=ExperimentExtra,
)


class ExperimentPatch(ExperimentUpdate):
    rerun_answers: bool = False


class ExperimentDetails(BaseModel):
    experiment_name: str
    dataset_name: str
    data: list[dict]

    model_config = ConfigDict(from_attributes=True)


#
# Experiment Set
#


class GridCV(BaseModel):
    common_params: dict[str, Any]
    grid_params: dict[str, list[Any]]
    repeat: int = 1


class ExperimentSetBase(EgBaseModel):
    name: str
    readme: str | None = None

    model_config = ConfigDict(from_attributes=True)


class ExperimentSetCreate(ExperimentSetBase):
    experiments: list[ExperimentCreate] | None = None
    cv: GridCV | None = None

    def to_table_init(self, db: Session) -> dict:
        obj = self.recurse_table_init(db)

        if self.experiments is not None and self.cv is not None:
            raise SchemaError("Please, give either an expriments or a cv parameter but not both.")

        # Handle Experiments
        if self.cv is not None:
            experiments = []
            i = 0
            for experiment in build_param_grid(self.cv.common_params, self.cv.grid_params):
                for _ in range(self.cv.repeat):
                    experiment["name"] = f"{self.name}__{i}"
                    experiments.append(ExperimentCreate(**experiment).to_table_init(db))
                    i += 1
            obj["experiments"] = experiments
        elif self.experiments is None:
            obj.pop("experiments")

        return obj


class ExperimentSet(ExperimentSetBase):
    id: int
    created_at: datetime
    experiments: list[Experiment] | None


ExperimentSetUpdate = create_model(
    "ExperimentSetUpdate",
    **{
        field_name: (Optional[field.annotation], None)
        for field_name, field in ExperimentSet.__fields__.items()
    },
    __base__=ExperimentSet,
)


class ExperimentSetPatch(ExperimentSetUpdate):
    pass
