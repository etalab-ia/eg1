import logging
import traceback
from dataclasses import dataclass
from io import StringIO

import pandas as pd
from sqlalchemy import update

import api.crud as crud
import api.models as models
from api.clients import LlmClient
from api.db import SessionLocal
from api.metrics import metric_registry
from api.runners import MessageType, dispatch_tasks
from api.utils import Timer, run_with_timeout


@dataclass
class MessageAnswer:
    message_type: MessageType  # message router identifier
    exp_id: str  # the experiment reference
    model_id: str  # the model to generate with
    line_id: int  # the line number of the dataset
    query: str  # the name of the observation/metric to process


def generate_answer(message: dict):
    """Message is a MessageAnswer dict containing the necessary information to process data"""
    msg = MessageAnswer(**message)
    with SessionLocal() as db:
        print("+", end="", flush=True)
        exp = crud.get_experiment(db, msg.exp_id)
        model = crud.get_model(db, msg.model_id)
        sampling_params = model.sampling_params or {}
        extra_params = model.extra_params or {}
        sampling_params_plus = sampling_params | extra_params
        answer = None
        error_msg = None
        try:
            # Generate answer
            # --
            messages = [{"role": "user", "content": msg.query}]
            if model.prompt_system:
                messages = [{"role": "system", "content": model.prompt_system}] + messages
            aiclient = LlmClient(base_url=model.base_url, api_key=model.api_key)
            with Timer() as timer:
                result = aiclient.generate(
                    model=model.name, messages=messages, **sampling_params_plus
                )
            answer = result.choices[0].message.content
            # Try to set the RAG metadata
            # @DEBUG: Is subject to change to align with albert-api...
            retrieval_context = None
            if hasattr(result, "rag_context"):
                refs = result.rag_context.references
                # @DEBUG: references are hash/is of chunks here.
                #         To get the content of the hash we need to query the api franceservices.../api/v2/indexes
                #         which need authentication. But authentication is painfull atm in this api. I will eventually make a PR to get the chunk content instead of the hashes...

            # Upsert answer
            crud.upsert_answer(
                db,
                exp.id,
                msg.line_id,
                dict(
                    answer=answer,
                    execution_time=timer.execution_time,
                    nb_tokens_prompt=result.usage.prompt_tokens,
                    nb_tokens_completion=result.usage.completion_tokens,
                    retrieval_context=retrieval_context,
                ),
            )

        except Exception as e:
            error_msg = "Generation failed with error: %s" % e
            logging.debug(traceback.print_exc())
            logging.error(error_msg)

        finally:
            # Ensure atomic transaction
            stmt = (
                update(models.Experiment)
                .where(models.Experiment.id == exp.id)
                .values(
                    num_try=models.Experiment.num_try + 1,
                    num_success=models.Experiment.num_success + (1 if answer else 0),
                )
                .returning(models.Experiment)
            )
            exp = db.execute(stmt).scalars().one()
            db.commit()

            if error_msg:
                crud.upsert_answer(db, exp.id, msg.line_id, dict(error_msg=error_msg))

        # Check if all the answer have been generated.
        if exp.num_try >= exp.dataset.size:
            dispatch_tasks(db, exp, MessageType.observation)


@dataclass
class MessageObservation:
    message_type: MessageType  # message router identifier
    exp_id: str  # the experiment reference
    line_id: int  # the line number of the dataset
    metric_name: str  # the name of the observation/metric to process
    output: str  # The actual output of the model
    output_true: str | None = None  # The ground truth output


def generate_observation(message: dict):
    """Message is a MessageObservation dict containing the necessary information to process data"""
    msg = MessageObservation(**message)
    with SessionLocal() as db:
        print(".", end="", flush=True)
        result = crud.get_result(db, experiment_id=msg.exp_id, metric_name=msg.metric_name)
        answer = crud.get_answer(db, experiment_id=msg.exp_id, num_line=msg.line_id)
        score = None
        observation = None
        error_msg = None
        metadata = {}
        if answer:  # answer.answer == msg.output
            metadata["generation_time"] = answer.execution_time
            metadata["nb_tokens_prompt"] = answer.nb_tokens_prompt
            metadata["nb_tokens_completion"] = answer.nb_tokens_completion
            metadata["retrieval_context"] = answer.retrieval_context
        try:
            # Generate observation/metric
            # --
            # Get the metric from registry
            metric = metric_registry.get_metric(msg.metric_name)
            metric_fun = metric_registry.get_metric_function(msg.metric_name)
            metric_params = {"metadata": metadata}
            if not metric_fun:
                raise ValueError(f"Metric {msg.metric_name} not found for experiment {msg.exp_id}")
            for require in metric.require:
                # Add extra inputs required by the metric
                if require in ["output", "output_true"]:
                    if not getattr(msg, require):
                        raise ValueError(
                            f"The metric {msg.metric_name} require a non null {require} value."
                        )
                    continue
                dataset = result.experiment.dataset
                df = pd.read_json(StringIO(dataset.df))
                try:
                    metric_params[require] = df.iloc[msg.line_id][require]
                except KeyError:
                    # If the required param is not in the dataset,
                    # try to get it from the metadata context.
                    metric_params[require] = metadata.get(require)

                if not metric_params[require]:
                    raise ValueError(
                        f"The metric {msg.metric_name} require a non null {require} value."
                    )

            # Compute metric
            with Timer() as timer:
                metric_result = metric_fun(msg.output, msg.output_true, **metric_params)
                # metric_result = run_with_timeout(
                #     metric_fun, 300, msg.output, msg.output_true, **metric_params
                # )
            if isinstance(metric_result, tuple):
                score, observation = metric_result
            else:
                score = metric_result

            # Upsert obsevation
            crud.upsert_observation(
                db,
                result.id,
                msg.line_id,
                dict(
                    observation=observation, score=score, execution_time=int(timer.execution_time)
                ),
            )

        except Exception as e:
            error_msg = f"Observation {msg.metric_name} failed with error: %s" % e
            logging.debug(traceback.print_exc())
            logging.error(error_msg)
        finally:
            # Ensure atomic transaction
            stmt = (
                update(models.Result)
                .where(models.Result.id == result.id)
                .values(
                    num_try=models.Result.num_try + 1,
                    num_success=models.Result.num_success + (1 if score is not None else 0),
                )
                .returning(models.Result)
            )
            result = db.execute(stmt).scalars().one()
            db.commit()

            if error_msg:
                crud.upsert_observation(db, result.id, msg.line_id, dict(error_msg=error_msg))

        # Check if all the answer have been generated.
        if result.num_try >= result.experiment.dataset.size:
            result = crud.update_result(db, result.id, dict(metric_status="finished"))
            print("x", end="", flush=True)
            if all(r.metric_status == "finished" for r in result.experiment.results):
                crud.update_experiment(db, msg.exp_id, dict(experiment_status="finished"))
                print("$", end="", flush=True)


def process_task(message: dict):
    """Route and process message"""
    match message["message_type"]:
        case MessageType.answer:
            task = generate_answer
        case MessageType.observation:
            task = generate_observation
        case _:
            raise NotImplementedError("Message type Unknown")

    return task(message)
