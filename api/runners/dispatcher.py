from enum import Enum
from io import StringIO

import pandas as pd
import zmq

import api.crud as crud
import api.schemas as schemas
from api.metrics import metric_registry


class MessageType(str, Enum):
    answer = "answers"  # Ask to generate an answer
    observation = "observations"  # Ask to generate an observation

    # @CODEFACTOR: for observation message, it is actually useless to pass
    # output and output_true as the df is loaded anyway to load the "require" fields...


def dispatch_tasks(db, db_exp, message_type: MessageType):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://localhost:5555")

    if message_type == MessageType.answer:
        # Generate answer
        # --
        # iterate the dataset
        crud.update_experiment(db, db_exp.id, dict(experiment_status="running_answers", num_try=0))
        df = pd.read_json(StringIO(db_exp.dataset.df))
        for num_line, row in df.iterrows():
            socket.send_json(
                {
                    "message_type": MessageType.answer,
                    "exp_id": db_exp.id,
                    "model_id": db_exp.model.id,
                    "line_id": num_line,
                    "query": row["query"],
                }
            )

        # The runner will check when all answers are finished
        # and follow with the observations if needed.
    elif message_type == MessageType.observation:
        # Generate observation
        # --
        # iterate metrics and dataset
        needs_output = any(  # to see if better to add a new table attribute to store this...
            "output" in metric_registry.get_metric(r.metric_name).require for r in db_exp.results
        )
        if len(db_exp.answers) == 0:
            raise NotImplementedError(
                "No answers available to generate observations for this experiment: %s" % db_exp.id
            )
        df = pd.read_json(StringIO(db_exp.dataset.df))
        crud.update_experiment(db, db_exp.id, dict(experiment_status="running_metrics"))
        for result in db_exp.results:
            if result.metric_status != "pending":
                continue
            result.num_try = 0
            result.num_success = 0
            result.metric_status = "running"
            db.commit()
            for a in db_exp.answers:
                row = df.iloc[a.num_line]
                socket.send_json(
                    {
                        "message_type": MessageType.observation,
                        "exp_id": db_exp.id,
                        "line_id": a.num_line,
                        "metric_name": result.metric_name,
                        "output": a.answer,
                        "output_true": row.get("output_true"),
                    }
                )

    else:
        raise ValueError("Task unkown: %s" % message_type)

    socket.close()
    context.term()


def dispatch_retries(db, retry_runs: schemas.RetryRuns):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://localhost:5555")

    for expid in retry_runs.experiment_ids:
        db_exp = crud.get_experiment(db, expid)
        if db_exp is None:
            raise ValueError("should never happen: dprexpnotfound1")

        crud.update_experiment(
            db, db_exp.id, dict(experiment_status="running_answers", num_try=db_exp.num_success)
        )
        df = pd.read_json(StringIO(db_exp.dataset.df))
        for answer in db_exp.answers:
            if answer.answer and not answer.error_msg:  # @TODO: add a is_failed columns !
                continue

            socket.send_json(
                {
                    "message_type": MessageType.answer,
                    "exp_id": db_exp.id,
                    "model_id": db_exp.model.id,
                    "line_id": answer.num_line,
                    "query": df.iloc[answer.num_line]["query"],
                }
            )

    for resultid in retry_runs.result_ids:
        result = crud.get_result(db, resultid)
        db_exp = result.experiment
        crud.update_experiment(db, db_exp.id, dict(experiment_status="running_metrics"))
        result.num_try -= result.num_success
        result.metric_status = "running"
        db.commit()
        df = pd.read_json(StringIO(db_exp.dataset.df))
        for obs in result.observation_table:
            if obs.score is not None and not obs.error_msg:  # @TODO: add a is_failed columns !
                continue

            answer = crud.get_answer(db, experiment_id=db_exp.id, num_line=obs.num_line)
            if not answer:
                answer = df.iloc[obs.num_line].get("output")
            else:
                answer = answer.answer

            socket.send_json(
                {
                    "message_type": MessageType.observation,
                    "exp_id": db_exp.id,
                    "line_id": obs.num_line,
                    "metric_name": result.metric_name,
                    "output": answer,
                    "output_true": df.iloc[obs.num_line].get("output_true"),
                }
            )

    socket.close()
    context.term()
