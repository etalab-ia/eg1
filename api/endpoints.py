import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import api.crud as crud
import api.schemas as schemas
from api.db import get_db
from api.errors import CustomIntegrityError, SchemaError
from api.metrics import Metric, metric_registry
from api.runners import dispatch_retries, dispatch_tasks
from api.security import admin_only

router = APIRouter()


def _needs_output(db_exp):
    return not db_exp.dataset.has_output and any(
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


@router.delete("/experiment/{id}")
def delete_experiment(
    id: int,
    db: Session = Depends(get_db),
    tags=["experiments"],
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
def read_experiments(set_id: int | None = None, limit: int = 100, db: Session = Depends(get_db)):
    experiments = crud.get_experiments(db, set_id=set_id, limit=limit)

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
    description="Re-run failed runs.",
    tags=["experiment_set"],
)
def retry_runs(id: int, db: Session = Depends(get_db)):
    experimentset = crud.get_experimentset(db, id)
    if experimentset is None:
        raise HTTPException(status_code=404, detail="ExperimentSet not found")

    rr = schemas.RetryRuns(experiment_ids=[], result_ids=[])
    for exp in experimentset.experiments:
        if exp.experiment_status != "finished":
            continue

        if exp.num_try != exp.num_success and _needs_output(exp):
            rr.experiment_ids.append(exp.id)
            continue

        for result in exp.results:
            if result.metric_status != "finished":
                continue

            if result.num_try != result.num_success:
                rr.result_ids.append(result.id)

    dispatch_retries(db, rr)
    return rr


#
# LeaderBoard
#


@router.get("/leaderboard", response_model=schemas.Leaderboard, tags=["leaderboard"])
def read_leaderboard(
    metric_name: str = "judge_notator",
    limit: int = 100,
    db: Session = Depends(get_db)
):
    return crud.get_leaderboard(db, metric_name=metric_name, limit=limit)
