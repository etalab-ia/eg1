from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, desc, and_
from sqlalchemy.exc import IntegrityError

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


def get_dataset(db: Session, dataset_id: int) -> models.Dataset | None:
    return db.query(models.Dataset).filter(models.Dataset.id == dataset_id).first()


def get_model(db: Session, model_id: int) -> models.Model | None:
    return db.query(models.Model).filter(models.Model.id == model_id).first()


def update_dataset(
    db: Session, dataset_id: int, dataset_update: schemas.DatasetUpdate | dict
) -> models.Dataset | None:
    if isinstance(dataset_update, dict):
        dataset_update = schemas.DatasetUpdate(**dataset_update)

    db_dataset = db.query(models.Dataset).get(dataset_id)
    if db_dataset is None:
        return None
    # Update fields
    for key, value in vars(dataset_update).items():
        setattr(db_dataset, key, value) if value else None
    db.commit()
    db.refresh(db_dataset)
    return db_dataset


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
    experiment_id: int | None = None,
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
    experiment_id: int | None = None,
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

    db_result = db.query(models.Result).get(result_id)
    if db_result is None:
        return None
    # Update fields
    for key, value in vars(result_update).items():
        setattr(db_result, key, value) if value else None
    db.commit()
    db.refresh(db_result)
    return db_result


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


def get_experiments(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    backward: bool = False,
    set_id: int | None = None,
    orphan: bool = False,
) -> list[models.Experiment]:
    limit = min(limit, 100)
    query = db.query(models.Experiment).filter_by(is_archived=False)
    if backward:
        query = query.order_by(models.Experiment.id.desc())
    else:
        query = query.order_by(models.Experiment.id.asc())

    if set_id:
        query = query.filter(models.Experiment.experiment_set_id == set_id)
    if orphan:
        query = query.filter(models.Experiment.experiment_set_id == None)
    return query.offset(skip).limit(limit).options(joinedload(models.Experiment.results)).all()


def update_experiment(
    db: Session, experiment_id: int, experiment_update: schemas.ExperimentUpdate | dict
) -> models.Experiment | None:
    if isinstance(experiment_update, dict):
        experiment_update = schemas.ExperimentUpdate(**experiment_update)

    db_exp = db.query(models.Experiment).get(experiment_id)
    if db_exp is None:
        return None
    # Update fields
    for key, value in vars(experiment_update).items():
        setattr(db_exp, key, value) if value else None
        # If experiment_status is set to finished, update all result status too.
        if key == "experiment_status" and value == schemas.ExperimentStatus.finished:
            for result in db_exp.results:
                result.metric_status = schemas.MetricStatus.finished
    db.commit()
    db.refresh(db_exp)
    return db_exp


def remove_experiment(db: Session, experiment_id: int) -> bool:
    db_exp = db.query(models.Experiment).get(experiment_id)
    if db_exp is None:
        return False

    # Delete linked results
    for result in db_exp.results:
        db.delete(result)

    # Delete linked answers
    for answer in db_exp.answers:
        db.delete(answer)

    db.delete(db_exp)
    db.commit()
    # update_db_exp(db, experiment_id, dict(is_archived=True))
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

    db_expset = db.query(models.ExperimentSet).get(experimentset_id)
    if db_expset is None:
        return None
    # Update fields
    for key, value in vars(experimentset_update).items():
        setattr(db_expset, key, value) if value else None
    db.commit()
    db.refresh(db_expset)
    return db_expset


def remove_experimentset(db: Session, experimentset_id: int) -> bool:
    db_expset = db.query(models.ExperimentSet).get(experimentset_id)
    if db_expset is None:
        return False

    for experiment in db_expset.experiments:
        remove_experiment(db, experiment.id)
    db.delete(db_expset)
    db.commit()
    return True


#
# Runner Crud
#


def upsert_answer(db: Session, experiment_id: int, num_line: int, answer: dict) -> models.Answer:
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
    db: Session, result_id: int, num_line: int, observation: dict
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


#
# LeaderBoard
#


def get_leaderboard(
    db: Session, metric_name: str = "judge_notator", dataset_name: str = None, limit: int = 100
):
    # Filter Experiment by dataset and metric
    main_metric_subquery = (
        db.query(
            models.Result.experiment_id, func.max(models.ObservationTable.score).label("main_score")
        )
        .join(models.ObservationTable)
        .filter(models.Result.metric_name == metric_name)
        .group_by(models.Result.experiment_id)
        .subquery()
    )

    # Query
    query = (
        db.query(
            models.Experiment.id.label("experiment_id"),
            models.Model.name.label("model_name"),
            models.Dataset.name.label("dataset_name"),
            main_metric_subquery.c.main_score.label("main_metric_score"),
            models.Model.sampling_params,
            models.Model.extra_params,
        )
        .join(models.Model)
        .join(models.Dataset)
        .join(main_metric_subquery, models.Experiment.id == main_metric_subquery.c.experiment_id)
    )

    if dataset_name:
        query = query.filter(models.Dataset.name == dataset_name)

    query = query.order_by(desc("main_metric_score")).limit(limit)

    # Execute
    results = query.all()

    # Fetch result in leaderboard
    entries = []
    for result in results:
        other_metrics = (
            db.query(models.Result.metric_name, models.ObservationTable.score)
            .join(models.ObservationTable)
            .filter(
                and_(
                    models.Result.experiment_id == result.experiment_id,
                    models.Result.metric_name != metric_name,
                )
            )
            .all()
        )

        entry = schemas.LeaderboardEntry(
            experiment_id=result.experiment_id,
            model_name=result.model_name,
            dataset_name=result.dataset_name,
            main_metric_score=result.main_metric_score,
            other_metrics={metric: score for metric, score in other_metrics},
            sampling_param={k: str(v) for k, v in (result.sampling_params or {}).items()},
            extra_param={k: str(v) for k, v in (result.extra_params or {}).items()},
        )
        entries.append(entry)

    return schemas.Leaderboard(entries=entries)


#
# Products
#


def create_product(db: Session, product: schemas.ProductCreate) -> models.Product:
    existing_product = db.query(models.Product).filter(models.Product.name == product.name).first()
    if existing_product:
        raise ValueError(f"Product with name '{product.name}' already exists")
    db_product = models.Product(**product.dict())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product

def create_products(db: Session, products: list[schemas.ProductCreate]) -> list[models.Product]:
    db_products = [models.Product(**product.dict()) for product in products]
    db.add_all(db_products)
    db.commit()
    for db_product in db_products:
        db.refresh(db_product)
    return db_products

def get_products(db: Session) -> list[models.Product]:
    return db.query(models.Product).all()

def get_product(db: Session, product_id: int) -> models.Product | None:
    return db.query(models.Product).filter(models.Product.id == product_id).first()

def create_product_dataset(db: Session, product_dataset: schemas.ProductDatasetCreate) -> models.ProductDataset:
    db_product_dataset = models.ProductDataset(**product_dataset.dict())
    try:
        db.add(db_product_dataset)
        db.commit()
        db.refresh(db_product_dataset)
        db.refresh(db_product_dataset, attribute_names=['dataset'])
        return db_product_dataset
    except IntegrityError:
        db.rollback()
        raise ValueError(f"ProductDataset with product_id {product_dataset.product_id} and dataset_id {product_dataset.dataset_id} already exists")

def get_product_datasets(db: Session) -> list[models.ProductDataset]:
    return db.query(models.ProductDataset).options(
        joinedload(models.ProductDataset.dataset)
    ).all()

def update_product_dataset(db: Session, product_dataset_id: int, update_data: schemas.ProductDatasetUpdate) -> models.ProductDataset | None:
    db_product_dataset = db.query(models.ProductDataset).filter(models.ProductDataset.id == product_dataset_id).first()
    if db_product_dataset:
        db_product_dataset.evaluation_metrics = update_data.evaluation_metrics
        db.commit()
        db.refresh(db_product_dataset)
        # Charger explicitement le dataset associé
        db.refresh(db_product_dataset, attribute_names=['dataset'])
    return db_product_dataset


#
# LOCUST
#


def create_locustrun(db: Session, run: schemas.LocustRunCreate) -> models.LocustRun:
    run = run.to_table_init(db) if isinstance(run, schemas.EgBaseModel) else run
    db_run = create_object_from_dict(db, models.LocustRun, run)
    db.add(db_run)
    db.commit()
    db.refresh(db_run)
    return db_run


def get_locustrun(db: Session, run_id: int) -> models.LocustRun | None:
    return db.query(models.LocustRun).filter(models.LocustRun.id == run_id).first()


def get_locustruns(
    db: Session, skip: int = 0, limit: int = 100, backward: bool = False
) -> list[models.LocustRun]:
    query = db.query(models.LocustRun)
    if backward:
        query = query.order_by(models.LocustRun.id.desc())
    else:
        query = query.order_by(models.LocustRun.id.asc())
    return query.offset(skip).limit(limit).all()
