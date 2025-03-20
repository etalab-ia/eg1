import re

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import eg1.api.crud as crud
import eg1.api.schemas as schemas
from eg1.api.db import get_db
from eg1.api.errors import CustomIntegrityError, SchemaError
from eg1.api.logger import logger
from eg1.api.metrics import Metric, metric_registry
from eg1.runners import dispatch_retries, dispatch_tasks
from eg1.api.security import admin_only

router = APIRouter()


def _needs_output(db_exp):
    return not db_exp.model.has_raw_output and any(
        "output" in metric_registry.get_metric(r.metric_name).require for r in db_exp.results
    )


#
# Datasets
#


@router.post("/dataset", response_model=schemas.Dataset, tags=["datasets"])
def create_dataset(dataset: schemas.DatasetCreate, db: Session = Depends(get_db)):
    try:
        db_dataset = crud.create_dataset(db, dataset)
        return db_dataset
    except (SchemaError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


@router.get("/datasets", response_model=list[schemas.Dataset], tags=["datasets"])
def read_datasets(db: Session = Depends(get_db)):
    return crud.get_datasets(db)


@router.get(
    "/dataset/{id}", response_model=schemas.Dataset | schemas.DatasetFull, tags=["datasets"]
)
def read_dataset(id: int, with_df: bool = False, db: Session = Depends(get_db)):
    dataset = crud.get_dataset(db, id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if with_df:
        return schemas.DatasetFull.from_orm(dataset)

    return schemas.Dataset.from_orm(dataset)


@router.get("/dataset", response_model=schemas.Dataset | schemas.DatasetFull, tags=["datasets"])
def read_dataset_by_query(
    id: str | None = None,
    name: str | None = None,
    with_df: bool = False,
    db: Session = Depends(get_db),
):
    dataset = None
    if name:
        dataset = crud.get_dataset_by_name(db, name)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset with given name not found")
    if id:
        dataset = crud.get_dataset(db, id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset:
        if with_df:
            return schemas.DatasetFull.from_orm(dataset)

        return schemas.Dataset.from_orm(dataset)

    raise HTTPException(status_code=400, detail="No query parameters provided")


@router.patch("/dataset/{id}", response_model=schemas.Dataset, tags=["datasets"])
def patch_dataset(id: int, dataset_patch: schemas.DatasetPatch, db: Session = Depends(get_db)):
    db_dataset = crud.update_dataset(db, id, dataset_patch)
    if db_dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return db_dataset


#
# Metrics
#


@router.get("/metrics", response_model=list[Metric], tags=["metrics"])
def read_metrics(db: Session = Depends(get_db)):
    return crud.get_metrics(db)


#
# Experiments
#


@router.post(
    "/experiment",
    response_model=schemas.Experiment,
    description="Launch an experiment. If a model is given, it will be use to generate the model output (answer), otherwise it will use the `output` column of the given dataset.",
    tags=["experiments"],
)
def create_experiment(experiment: schemas.ExperimentCreate, db: Session = Depends(get_db)):
    try:
        db_exp = crud.create_experiment(db, experiment)
        if _needs_output(db_exp):
            dispatch_tasks(db, db_exp, "answers")
        else:
            dispatch_tasks(db, db_exp, "observations")

        return db_exp

    except (SchemaError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


@router.patch(
    "/experiment/{id}",
    response_model=schemas.Experiment,
    description="Update an experiment. The given metrics will be added (or rerun) to the existing results for this experiments. Use rerun_answers if want to re-generate the answers/output.",
    tags=["experiments"],
)
def patch_experiment(
    id: int, experiment_patch: schemas.ExperimentPatch, db: Session = Depends(get_db)
):
    db_exp = crud.update_experiment(db, id, experiment_patch)
    if db_exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    elif db_exp.experiment_status not in [
        schemas.ExperimentStatus.pending,
        schemas.ExperimentStatus.finished,
    ]:
        raise HTTPException(
            status_code=400,
            detail=f"Experiment is running ({db_exp.experiment_status}), please try again later",
        )

    # Rerun experiment
    # --
    # Initialize metric results
    for metric in experiment_patch.metrics or []:
        result = crud.get_result(db, experiment_id=db_exp.id, metric_name=metric)
        if result:
            crud.update_result(db, result.id, dict(metric_status="pending"))
        else:
            result = schemas.ResultCreate(experiment_id=db_exp.id, metric_name=metric)
            crud.create_result(db, result)
    # Dispatch tasks
    if experiment_patch.rerun_answers and _needs_output(db_exp):
        dispatch_tasks(db, db_exp, "answers")
    elif experiment_patch.rerun_metrics:
        dispatch_tasks(db, db_exp, "observations")

    return db_exp


@router.delete(
    "/experiment/{id}",
    tags=["experiments"],
)
def delete_experiment(
    id: int,
    db: Session = Depends(get_db),
    admin_check=Depends(admin_only),
):
    if not crud.remove_experiment(db, id):
        raise HTTPException(status_code=404, detail="Experiment not found")
    return "ok"


@router.get(
    "/experiment/{id}",
    response_model=schemas.Experiment
    | schemas.ExperimentWithResults
    | schemas.ExperimentWithAnswers
    | schemas.ExperimentFull
    | schemas.ExperimentFullWithDataset,
    tags=["experiments"],
)
def read_experiment(
    id: int,
    with_results: bool = False,
    with_answers: bool = False,
    with_dataset: bool = False,
    db: Session = Depends(get_db),
):
    experiment = crud.get_experiment(db, id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if with_dataset:
        return schemas.ExperimentFullWithDataset.from_orm(experiment)
    elif with_answers and with_results:
        return schemas.ExperimentFull.from_orm(experiment)
    elif with_results:
        return schemas.ExperimentWithResults.from_orm(experiment)
    elif with_answers:
        return schemas.ExperimentWithAnswers.from_orm(experiment)

    return schemas.Experiment.from_orm(experiment)


@router.get(
    "/experiments",
    response_model=list[schemas.ExperimentWithResults],
    tags=["experiments"],
)
def read_experiments(
    skip: int = 0,
    limit: int = 100,
    backward: bool = True,
    set_id: int | None = None,
    orphan: bool = True,
    db: Session = Depends(get_db),
):
    experiments = crud.get_experiments(
        db, skip=skip, limit=limit, backward=backward, set_id=set_id, orphan=orphan
    )

    if not experiments:
        raise HTTPException(status_code=404, detail="No experiments found")

    return experiments


#
# Experiment Sets
#


@router.post(
    "/experiment_set",
    response_model=schemas.ExperimentSet,
    tags=["experiment_set"],
)
def create_experimentset(experimentset: schemas.ExperimentSetCreate, db: Session = Depends(get_db)):
    try:
        db_expset = crud.create_experimentset(db, experimentset)
        for db_exp in db_expset.experiments:
            if _needs_output(db_exp):
                dispatch_tasks(db, db_exp, "answers")
            else:
                dispatch_tasks(db, db_exp, "observations")

        return db_expset
    except (SchemaError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


@router.patch(
    "/experiment_set/{id}",
    response_model=schemas.ExperimentSet,
    description="Update an experimentset: New experiments will be added to the runner queue.",
    tags=["experiment_set"],
)
def patch_experimentset(
    id: int, experimentset_patch: schemas.ExperimentSetPatch, db: Session = Depends(get_db)
):
    db_expset = crud.update_experimentset(db, id, experimentset_patch)
    if db_expset is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    expset = experimentset_patch.to_table_init(db)
    for experiment in expset.get("experiments") or []:
        experiment["experiment_set_id"] = id
        # Respect the unique constraint for auto-naming experiment !
        # -> add an increment suffix to the experiment name
        if re.search(r"__\d+$", experiment["name"]):
            parts = experiment["name"].split("__")
            parts[-1] = str(int(parts[-1]) + len(db_expset.experiments))
            if parts[0] == "None":
                parts[0] = db_expset.name
            experiment["name"] = "__".join(parts)
        db_exp = crud.create_experiment(db, experiment)
        if _needs_output(db_exp):
            dispatch_tasks(db, db_exp, "answers")
        else:
            dispatch_tasks(db, db_exp, "observations")

    return db_expset


@router.get(
    "/experiment_sets",
    response_model=list[schemas.ExperimentSet],
    tags=["experiment_set"],
)
def read_experimentsets(db: Session = Depends(get_db)):
    experimentsets = crud.get_experimentsets(db)
    if experimentsets is None:
        raise HTTPException(status_code=404, detail="ExperimentSets not found")
    return experimentsets
    # return [schemas.ExperimentSet.from_orm(x) for x in experimentsets]


@router.get(
    "/experiment_set/{id}",
    response_model=schemas.ExperimentSet,
    tags=["experiment_set"],
)
def read_experimentset(id: int, db: Session = Depends(get_db)):
    experimentset = crud.get_experimentset(db, id)
    if experimentset is None:
        raise HTTPException(status_code=404, detail="ExperimentSet not found")
    return experimentset


@router.delete(
    "/experiment_set/{id}",
    tags=["experiment_set"],
)
def delete_experimentset(id: int, db: Session = Depends(get_db), admin_check=Depends(admin_only)):
    if not crud.remove_experimentset(db, id):
        raise HTTPException(status_code=404, detail="ExperimentSet not found")
    return "ok"


@router.post(
    "/retry/experiment_set/{id}",
    response_model=schemas.RetryRuns,
    description="Re-run failed runs and continue unfinished answers or metrics.",
    tags=["experiment_set"],
)
def retry_runs(
    id: int,
    force: bool = Query(
        default=False,
        description="Force retry of all unfinished runs, by resetting their status to pending. <!> Warning this can cause incoherent num_try/num_success value if another runner work on the same experiments.",
    ),
    db: Session = Depends(get_db),
):
    experimentset = crud.get_experimentset(db, id)
    if experimentset is None:
        raise HTTPException(status_code=404, detail="ExperimentSet not found")

    rr = schemas.RetryRuns(
        experiment_ids=[], result_ids=[], unfinished_experiment_ids=[], unfinished_result_ids=[]
    )

    # Retry failed runs (answers and metrics)
    for exp in experimentset.experiments:
        if exp.experiment_status != "finished" and not force:
            continue

        if exp.num_try != exp.num_success and _needs_output(exp):
            rr.experiment_ids += [exp.id]
            continue

        for result in exp.results:
            if result.metric_status != "finished" and not force:
                continue

            if result.num_try != result.num_success:
                rr.result_ids += [result.id]
                if force:
                    crud.update_result(db, result.id, dict(metric_status="finished"))

    # Unfinished business
    for exp in experimentset.experiments:
        if exp.experiment_status != "finished" and not force:
            continue

        expected_output_len = exp.dataset.size
        actual_output_len = exp.num_try
        if actual_output_len != expected_output_len:
            rr.unfinished_experiment_ids += [exp.id]
            continue

        for result in exp.results:
            if result.metric_status != "finished" and not force:
                continue

            expected_output_len = exp.dataset.size
            actual_output_len = result.num_try
            if actual_output_len != expected_output_len:
                rr.unfinished_result_ids += [result.id]
                if force:
                    crud.update_result(db, result.id, dict(metric_status="finished"))

    if (
        force and len(set(rr.experiment_ids + rr.result_ids +rr.unfinished_experiment_ids + rr.unfinished_result_ids)) == 0
    ):  # fmt: skip
        for exp in experimentset.experiments:
            crud.update_experiment(db, exp.id, dict(experiment_status="finished"))

    dispatch_retries(db, rr)
    return rr


#
# LeaderBoard
#


@router.get("/leaderboard", response_model=schemas.Leaderboard, tags=["leaderboard"])
def read_leaderboard(
    metric_name: str = "judge_notator",
    dataset_name: str = None,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    return crud.get_leaderboard(db, metric_name=metric_name, dataset_name=dataset_name, limit=limit)


#
# LOCUST
#


@router.post(
    "/locust",
    response_model=schemas.LocustRun,
    description="""Save a locust run.

    To format the locust CSV as a dataframe, here is how you must convert it to a dataframe:
    ```
    import pandas as pd
    import requests
    stats_df = pd.read_csv("stats.csv").to_json()

    # Then you can just pass the data in the POST request along the other parameters.
    ```
    """,
    tags=["locust"],
)
def create_locustrun(run: schemas.LocustRunCreate, db: Session = Depends(get_db)):
    try:
        db_run = crud.create_locustrun(db, run)
        return db_run

    except (SchemaError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


@router.get(
    "/locust/runs",
    response_model=list[schemas.LocustRun],
    description="Get the list of locust runs",
    tags=["locust"],
)
def get_locustruns(
    skip: int = 0, limit: int = 100, backward: bool = True, db: Session = Depends(get_db)
):
    try:
        db_runs = crud.get_locustruns(db, skip=skip, limit=limit, backward=backward)
        return db_runs

    except (SchemaError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


@router.get(
    "/locust/{id}",
    response_model=schemas.LocustRunFull,
    description="Get locust run with data.",
    tags=["locust"],
)
def get_locustrun(run_id: int, db: Session = Depends(get_db)):
    try:
        db_run = crud.get_locustrun(db, run_id)
        return db_run

    except (SchemaError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e
