from sqlalchemy.orm import joinedload, Session

import api.models as models
import api.schemas as schemas
from api.metrics import Metric, metric_registry
from api.models import create_object_from_dict

#
# Datasets
#


def create_dataset(db: Session, dataset: schemas.DatasetCreate) -> models.Dataset:
    dataset = dataset.to_table_init(db) if isinstance(dataset, schemas.EgBaseModel) else dataset
    db_dataset = create_object_from_dict(db, models.Dataset, dataset)
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset


def get_datasets(db: Session) -> list[models.Dataset]:
    return db.query(models.Dataset).all()


def get_dataset(db: Session, dataset_id: str) -> models.Dataset | None:
    return db.query(models.Dataset).filter(models.Dataset.name == dataset_id).first()


def get_model(db: Session, model_id: int) -> models.Model | None:
    return db.query(models.Model).filter(models.Model.id == model_id).first()


#
# Metrics
#


def get_metrics(db: Session) -> list[Metric]:
    return list(metric_registry.get_metrics())


#
# Results
#


def get_answer(
    db: Session,
    answer_id: int | None = None,
    experiment_id: str | None = None,
    num_line: int | None = None,
) -> models.Answer | None:
    if answer_id:
        return db.query(models.Answer).filter(models.Answer.id == answer_id).first()
    elif experiment_id and num_line is not None:
        return (
            db.query(models.Answer)
            .filter_by(experiment_id=experiment_id, num_line=num_line)
            .first()
        )
    else:
        raise ValueError("Should give at list an answer_id or a experiment_id/num_line couple.")


def get_result(
    db: Session,
    result_id: int | None = None,
    experiment_id: str | None = None,
    metric_name: str | None = None,
) -> models.Result | None:
    if result_id:
        return db.query(models.Result).filter(models.Result.id == result_id).first()
    elif experiment_id and metric_name:
        # TODO: unique by name vs configurable metric (always acceded is by result_id)
        return (
            db.query(models.Result)
            .filter_by(experiment_id=experiment_id, metric_name=metric_name)
            .first()
        )
    else:
        raise ValueError(
            "Should give at list an result_id or a couple experiment_id/metric_name couple."
        )


def create_result(db: Session, result: schemas.ResultCreate) -> models.Result:
    result = result.to_table_init(db) if isinstance(result, schemas.EgBaseModel) else result
    db_result = create_object_from_dict(db, models.Result, result)
    db.add(db_result)
    db.commit()
    db.refresh(db_result)
    return db_result


def update_result(
    db: Session, result_id: int, result_update: schemas.ResultUpdate | dict
) -> models.Result | None:
    if isinstance(result_update, dict):
        result_update = schemas.ResultUpdate(**result_update)

    result = db.query(models.Result).get(result_id)
    if result is None:
        return None
    # Update fields
    for var, value in vars(result_update).items():
        setattr(result, var, value) if value else None
    db.commit()
    db.refresh(result)
    return result


#
# Experiments
#


def create_experiment(db: Session, experiment: schemas.ExperimentCreate) -> models.Experiment:
    experiment = (
        experiment.to_table_init(db) if isinstance(experiment, schemas.EgBaseModel) else experiment
    )
    db_exp = create_object_from_dict(db, models.Experiment, experiment)
    db.add(db_exp)
    db.commit()
    db.refresh(db_exp)
    return db_exp


def get_experiment(db: Session, experiment_id: int) -> models.Experiment | None:
    return db.query(models.Experiment).get(experiment_id)


def get_experiments(db: Session, set_id: int | None = None) -> list[models.Experiment]:
    query = db.query(models.Experiment).filter_by(is_archived=False)
    if set_id:
        query = query.filter(models.Experiment.experiment_set_id == set_id).order_by(models.Experiment.created_at.desc())
    return query.limit(100).options(joinedload(models.Experiment.results)).all()


def update_experiment(
    db: Session, experiment_id: int, experiment_update: schemas.ExperimentUpdate | dict
) -> models.Experiment | None:
    if isinstance(experiment_update, dict):
        experiment_update = schemas.ExperimentUpdate(**experiment_update)

    experiment = db.query(models.Experiment).get(experiment_id)
    if experiment is None:
        return None
    # Update fields
    for var, value in vars(experiment_update).items():
        setattr(experiment, var, value) if value else None
    db.commit()
    db.refresh(experiment)
    return experiment


def remove_experiment(db: Session, experiment_id: int) -> bool:
    experiment = db.query(models.Experiment).get(experiment_id)
    if experiment is None:
        return False
    # db.delete(experiment)
    # db.commit()
    # return True
    update_experiment(db, experiment_id, dict(is_archived=True))
    return True


#
# Experiment Sets
#


def create_experimentset(
    db: Session, experimentset: schemas.ExperimentSetCreate
) -> models.ExperimentSet:
    experimentset = (
        experimentset.to_table_init(db)
        if isinstance(experimentset, schemas.EgBaseModel)
        else experimentset
    )
    db_expset = create_object_from_dict(db, models.ExperimentSet, experimentset)
    db.add(db_expset)
    db.commit()
    db.refresh(db_expset)
    return db_expset


def get_experimentsets(db: Session) -> list[models.ExperimentSet]:
    query = db.query(models.ExperimentSet)
    return query.all()


def get_experimentset(db: Session, experimentset_id: int) -> models.ExperimentSet | None:
    return db.query(models.ExperimentSet).get(experimentset_id)


def update_experimentset(
    db: Session, experimentset_id: int, experimentset_update: schemas.ExperimentSetUpdate | dict
) -> models.ExperimentSet | None:
    if isinstance(experimentset_update, dict):
        experimentset_update = schemas.ExperimentSetUpdate(**experimentset_update)

    experimentset = db.query(models.ExperimentSet).get(experimentset_id)
    if experimentset is None:
        return None
    # Update fields
    for var, value in vars(experimentset_update).items():
        setattr(experimentset, var, value) if value else None
    db.commit()
    db.refresh(experimentset)
    return experimentset


def remove_experimentset(db: Session, experimentset_id: int) -> bool:
    experimentset = db.query(models.ExperimentSet).get(experimentset_id)
    if experimentset is None:
        return False
    # db.delete(experimentset)
    # db.commit()
    # return True
    update_experimentset(db, experimentset_id, dict(is_archived=True))
    for exp_id in experimentset.experiments:
        update_experiment(db, exp_id, dict(is_archived=True))
    return True


#
# Runner Crud
#


def upsert_answer(db: Session, experiment_id: str, num_line: int, answer: dict) -> models.Answer:
    # Check if the record already exists
    db_answer = (
        db.query(models.Answer).filter_by(num_line=num_line, experiment_id=experiment_id).first()
    )

    if db_answer:
        # Update the existing record
        for k, v in answer.items():
            setattr(db_answer, k, v)
    else:
        # Insert a new record
        db_answer = models.Answer(experiment_id=experiment_id, num_line=num_line, **answer)
        db.add(db_answer)

    db.commit()
    db.refresh(db_answer)
    return db_answer


def upsert_observation(
    db: Session, result_id: str, num_line: int, observation: dict
) -> models.Answer:
    # Check if the record already exists
    db_observation = (
        db.query(models.ObservationTable).filter_by(num_line=num_line, result_id=result_id).first()
    )

    if db_observation:
        # Update the existing record
        for k, v in observation.items():
            setattr(db_observation, k, v)
    else:
        # Insert a new record
        db_observation = models.ObservationTable(
            result_id=result_id, num_line=num_line, **observation
        )
        db.add(db_observation)

    db.commit()
    db.refresh(db_observation)
    return db_observation
