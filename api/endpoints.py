from fastapi import APIRouter, Depends, HTTPException
import re
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import api.crud as crud
import api.schemas as schemas
from api.db import get_db
from api.errors import CustomIntegrityError, SchemaError
from api.metrics import Metric, metric_registry
from api.runners import dispatch_tasks

router = APIRouter()


#
# Datasets
#


@router.post("/dataset", response_model=schemas.Dataset)
def create_dataset(dataset: schemas.DatasetCreate, db: Session = Depends(get_db)):
    try:
        db_dataset = crud.create_dataset(db, dataset)
        return db_dataset
    except SchemaError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


@router.get("/datasets", response_model=list[schemas.Dataset])
def read_datasets(db: Session = Depends(get_db)):
    return crud.get_datasets(db)


@router.get("/dataset/{id}", response_model=schemas.Dataset | schemas.DatasetFull)
def read_dataset(id: int, with_df: bool = False, db: Session = Depends(get_db)):
    dataset = crud.get_dataset(db, id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if with_df:
        return schemas.DatasetFull.from_orm(dataset)

    return schemas.Dataset.from_orm(dataset)


#
# Metrics
#


@router.get("/metrics", response_model=list[Metric])
def read_metrics(db: Session = Depends(get_db)):
    return crud.get_metrics(db)


#
# Experiments
#


@router.post(
    "/experiment",
    response_model=schemas.Experiment,
    description="Launch an experiment. If a model is given, it will be use to generate the model output (answer), otherwise it will use the `output` column of the given dataset.",
)
def create_experiment(experiment: schemas.ExperimentCreate, db: Session = Depends(get_db)):
    try:
        db_exp = crud.create_experiment(db, experiment)
        needs_output = any(
            "output" in metric_registry.get_metric(m).require for m in experiment.metrics
        )
        if needs_output and not db_exp.dataset.has_output:
            dispatch_tasks(db, db_exp, "answers")
        else:
            dispatch_tasks(db, db_exp, "observations")

        return db_exp

    except SchemaError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


@router.patch(
    "/experiment/{id}",
    response_model=schemas.Experiment,
    description="Update an experiments. The given metrics will be added to the existing results for this experiments. Use rerun_answers if want to re-generate the answers/output.",
)
def patch_experiment(
    id: int, experiment_patch: schemas.ExperimentPatch, db: Session = Depends(get_db)
):
    experiment = crud.update_experiment(db, id, experiment_patch)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    elif experiment.experiment_status not in [
        schemas.ExperimentStatus.pending,
        schemas.ExperimentStatus.finished,
    ]:
        raise HTTPException(
            status_code=400,
            detail=f"Experiment is running ({experiment.experiment_status}), please try again later",
        )

    # Rerun experiment
    # --
    # Initialize metric results
    for metric in experiment_patch.metrics:
        result = crud.get_result(db, experiment_id=experiment.id, metric_name=metric)
        if result:
            crud.update_result(db, result.id, dict(metric_status="pending"))
        else:
            result = schemas.ResultCreate(experiment_id=experiment.id, metric_name=metric)
            crud.create_result(db, result)
    # Dispatch tasks
    if experiment.dataset.has_output or not experiment_patch.rerun_answers:
        dispatch_tasks(db, experiment, "observations")
    else:
        dispatch_tasks(db, experiment, "answers")

    return experiment


@router.delete("/experiment/{id}", status_code=204)
def delete_experiment(id: int, db: Session = Depends(get_db)):
    if not crud.remove_experiment(db, id):
        raise HTTPException(status_code=404, detail="Experiment not found")
    return


@router.get(
    "/experiment/{id}",
    response_model=schemas.Experiment | schemas.ExperimentWithResults | schemas.ExperimentFull,
)
def read_experiment(
    id: int, with_answers: bool = False, with_results: bool = False, db: Session = Depends(get_db)
):
    experiment = crud.get_experiment(db, id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if with_answers and with_results:
        return schemas.ExperimentFull.from_orm(experiment)
    elif with_results:
        return schemas.ExperimentWithResults.from_orm(experiment)
    elif with_answers:
        return schemas.ExperimentWithAnswers.from_orm(experiment)

    return schemas.Experiment.from_orm(experiment)


@router.get("/experiments", response_model=list[schemas.ExperimentWithResults])
def read_experiments(db: Session = Depends(get_db)):
    experiments = crud.get_experiments(db)
    
    if not experiments:
        raise HTTPException(status_code=404, detail="No experiments found")

    return experiments


@router.get("/experiments/{experiment_id}/details", response_model=schemas.ExperimentDetails)
def get_experiment_details(experiment_id: int, db: Session = Depends(get_db)):
    details = crud.get_experiment_details(db, experiment_id)
    if not details:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return details
    

#
# Experiment Sets
#


@router.post("/experiment_set", response_model=schemas.ExperimentSet)
def create_experimentset(experimentset: schemas.ExperimentSetCreate, db: Session = Depends(get_db)):
    try:
        db_expset = crud.create_experimentset(db, experimentset)
        for db_exp in db_expset.experiments:
            if db_exp.dataset.has_output:
                dispatch_tasks(db, db_exp, "observations")
            else:
                dispatch_tasks(db, db_exp, "answers")

        return db_expset
    except SchemaError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        return CustomIntegrityError.from_integrity_error(e.orig).to_http_response()
    except Exception as e:
        raise e


@router.patch(
    "/experiment_set/{id}",
    response_model=schemas.ExperimentSet,
    description="Update an experimentset: New experiment will be added to the run queue.",
)
def patch_experimentset(
    id: int, experimentset: schemas.ExperimentSetPatch, db: Session = Depends(get_db)
):
    expset = experimentset.to_table_init(db)
    db_experimentset = crud.update_experimentset(db, id, experimentset)
    if db_experimentset is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    for experiment in expset.get("experiments") or []:
        experiment["experiment_set_id"] = id
        # Respect the unique constraint for auto-naming experiment !
        if re.search(r"__\d+$", experiment["name"]):
            parts = experiment["name"].split("__")
            parts[-1] = str(int(parts[-1]) + len(db_experimentset.experiments))
            if parts[0] == "None":
                parts[0] = db_experimentset.name
            experiment["name"] = "__".join(parts)
        db_exp = crud.create_experiment(db, experiment)
        if db_exp.dataset.has_output:
            dispatch_tasks(db, db_exp, "observations")
        else:
            dispatch_tasks(db, db_exp, "answers")

    return db_experimentset


@router.get("/experiment_sets", response_model=list[schemas.ExperimentSet])
def read_experimentsets(db: Session = Depends(get_db)):
    experimentsets = crud.get_experimentsets(db)
    if experimentsets is None:
        raise HTTPException(status_code=404, detail="ExperimentSets not found")
    return experimentsets
    # return [schemas.ExperimentSet.from_orm(x) for x in experimentsets]


@router.get("/experiment_set/{id}", response_model=schemas.ExperimentSet)
def read_experimentset(id: int, db: Session = Depends(get_db)):
    experimentset = crud.get_experimentset(db, id)
    if experimentset is None:
        raise HTTPException(status_code=404, detail="ExperimentSet not found")
    return experimentset


@router.delete("/experiment_set/{id}", status_code=204)
def delete_experimentset(id: int, db: Session = Depends(get_db)):
    if not crud.remove_experimentset(db, id):
        raise HTTPException(status_code=404, detail="ExperimentSet not found")
    return
