"""
DAG Integrity Tests (Lab 5).

Перевіряє, що DAG-файли:
- завантажуються без помилок імпорту
- містять очікувані задачі
- мають правильну топологію залежностей
"""

import os

import pytest
from airflow.models import DagBag  # noqa: E402

DAGS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "dags")
EXPECTED_DAG_ID = "ml_training_pipeline"
EXPECTED_TASKS = [
    "check_data_availability",
    "prepare_data",
    "train_model",
    "evaluate_model",
    "check_model_quality",
    "register_model",
    "stop_pipeline",
]


@pytest.fixture(scope="module")
def dag_bag():
    """Завантажує всі DAG-и з папки dags/."""
    return DagBag(dag_folder=DAGS_FOLDER, include_examples=False)


def test_no_import_errors(dag_bag):
    """DAG-файли не повинні мати помилок імпорту."""
    assert len(dag_bag.import_errors) == 0, (
        f"DAG import errors found:\n"
        + "\n".join(
            f"  {path}: {err}"
            for path, err in dag_bag.import_errors.items()
        )
    )


def test_ml_pipeline_dag_exists(dag_bag):
    """DAG ml_training_pipeline має існувати в DagBag."""
    assert EXPECTED_DAG_ID in dag_bag.dags, (
        f"DAG '{EXPECTED_DAG_ID}' not found. "
        f"Available DAGs: {list(dag_bag.dags.keys())}"
    )


def test_all_expected_tasks_present(dag_bag):
    """Всі очікувані task_id мають бути присутні в DAG."""
    dag = dag_bag.get_dag(EXPECTED_DAG_ID)
    actual_task_ids = {task.task_id for task in dag.tasks}

    missing = set(EXPECTED_TASKS) - actual_task_ids
    assert not missing, (
        f"Missing tasks in DAG: {missing}\n"
        f"Actual tasks: {actual_task_ids}"
    )


def test_dag_has_no_cycles(dag_bag):
    """DAG не повинен містити циклів (ациклічність)."""
    dag = dag_bag.get_dag(EXPECTED_DAG_ID)
    # Airflow сам перевіряє ациклічність під час завантаження DAG.
    # Якщо DAG завантажився без помилок — граф ациклічний.
    assert dag is not None


def test_task_dependencies(dag_bag):
    """Перевіряє базову топологію пайплайну."""
    dag = dag_bag.get_dag(EXPECTED_DAG_ID)
    task_dict = {t.task_id: t for t in dag.tasks}

    # prepare_data залежить від check_data_availability
    prepare_upstream = {
        t.task_id for t in task_dict["prepare_data"].upstream_list
    }
    assert "check_data_availability" in prepare_upstream

    # train_model залежить від prepare_data
    train_upstream = {
        t.task_id for t in task_dict["train_model"].upstream_list
    }
    assert "prepare_data" in train_upstream

    # evaluate_model залежить від train_model
    eval_upstream = {
        t.task_id for t in task_dict["evaluate_model"].upstream_list
    }
    assert "train_model" in eval_upstream

    # check_model_quality залежить від evaluate_model
    branch_upstream = {
        t.task_id for t in task_dict["check_model_quality"].upstream_list
    }
    assert "evaluate_model" in branch_upstream


def test_schedule_interval(dag_bag):
    """DAG має встановлений розклад."""
    dag = dag_bag.get_dag(EXPECTED_DAG_ID)
    assert dag.schedule_interval is not None


def test_catchup_disabled(dag_bag):
    """catchup має бути False, щоб уникнути зайвих backfill-запусків."""
    dag = dag_bag.get_dag(EXPECTED_DAG_ID)
    assert dag.catchup is False, "DAG catchup must be False"
