"""
DAG: ml_training_pipeline
Оркеструє повний ML-пайплайн для класифікації шкірних уражень (HAM10000).

Кроки:
    1. check_data_availability  — FileSensor: перевіряє наявність CSV-файлу
    2. prepare_data             — BashOperator: запускає src/prepare.py
    3. train_model              — BashOperator: запускає src/train_pipeline.py
    4. evaluate_model           — PythonOperator: читає metrics.json → XCom
    5. check_model_quality      — BranchPythonOperator: accuracy >= поріг?
    6a. register_model          — PythonOperator: реєстрація в MLflow (Staging)
    6b. stop_pipeline           — BashOperator: лог про недостатню якість
"""

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.sensors.python import PythonSensor

# ---------------------------------------------------------------------------
# Конфігурація через змінні середовища (задаються в docker-compose)
# ---------------------------------------------------------------------------
PROJECT_PATH = os.environ.get("ML_PROJECT_PATH", "/opt/ml_project")
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python")
ACCURACY_THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "0.50"))

# ---------------------------------------------------------------------------
# Параметри DAG за замовчуванням
# ---------------------------------------------------------------------------
default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    description="HAM10000 ML pipeline: Prepare → Train → Evaluate → Register",
    schedule_interval="@daily",
    catchup=False,
    tags=["ml", "training", "ham10000"],
)

# ---------------------------------------------------------------------------
# Завдання 1: Sensor — перевірка наявності вхідних даних
# ---------------------------------------------------------------------------
def _check_data_exists():
    """Повертає True, якщо CSV-файл з даними існує."""
    data_path = os.path.join(PROJECT_PATH, "data", "raw", "hmnist_28_28_L.csv")
    exists = os.path.exists(data_path)
    if not exists:
        print(f"[check_data] File not found: {data_path}")
    return exists


check_data = PythonSensor(
    task_id="check_data_availability",
    python_callable=_check_data_exists,
    timeout=120,
    poke_interval=15,
    mode="poke",
    dag=dag,
)

# ---------------------------------------------------------------------------
# Завдання 2: Підготовка даних (dvc repro або прямий виклик prepare.py)
# ---------------------------------------------------------------------------
prepare_data = BashOperator(
    task_id="prepare_data",
    bash_command=(
        f"cd {PROJECT_PATH} && "
        f"{PYTHON_BIN} src/prepare.py "
        f"data/raw/hmnist_28_28_L.csv data/prepared"
    ),
    dag=dag,
)

# ---------------------------------------------------------------------------
# Завдання 3: Тренування моделі
# ---------------------------------------------------------------------------
train_model = BashOperator(
    task_id="train_model",
    bash_command=(
        f"cd {PROJECT_PATH} && "
        f"{PYTHON_BIN} src/train_pipeline.py"
    ),
    dag=dag,
)

# ---------------------------------------------------------------------------
# Завдання 4: Оцінка моделі — читаємо metrics.json та пушимо в XCom
# ---------------------------------------------------------------------------
def _evaluate_model(**kwargs):
    """Читає metrics.json та зберігає результати у XCom для наступних задач."""
    metrics_path = os.path.join(PROJECT_PATH, "metrics.json")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.json not found at {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print(f"[evaluate_model] Metrics: {metrics}")
    kwargs["ti"].xcom_push(key="metrics", value=metrics)
    return metrics


evaluate_model = PythonOperator(
    task_id="evaluate_model",
    python_callable=_evaluate_model,
    dag=dag,
)

# ---------------------------------------------------------------------------
# Завдання 5: Розгалуження — реєструємо або зупиняємо пайплайн
# ---------------------------------------------------------------------------
def _check_model_quality(**kwargs):
    """
    BranchPythonOperator: повертає task_id наступного завдання.
    Якщо accuracy >= поріг → register_model
    Інакше → stop_pipeline
    """
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="evaluate_model", key="metrics")

    if metrics is None:
        raise ValueError("No metrics found in XCom from evaluate_model task")

    accuracy = metrics.get("accuracy", 0.0)
    print(
        f"[check_model_quality] accuracy={accuracy:.4f}, "
        f"threshold={ACCURACY_THRESHOLD}"
    )

    if accuracy >= ACCURACY_THRESHOLD:
        print("Model quality PASSED — proceeding to registration.")
        return "register_model"

    print("Model quality FAILED — stopping pipeline.")
    return "stop_pipeline"


check_quality = BranchPythonOperator(
    task_id="check_model_quality",
    python_callable=_check_model_quality,
    dag=dag,
)

# ---------------------------------------------------------------------------
# Завдання 6a: Реєстрація моделі в MLflow Model Registry (stage=Staging)
# ---------------------------------------------------------------------------
def _register_model(**kwargs):
    """
    Реєструє модель у MLflow Model Registry зі стадією Staging.
    Використовує model.pkl та metrics.json, що були збережені train_pipeline.
    """
    import joblib
    import mlflow
    import mlflow.sklearn

    model_path = os.path.join(PROJECT_PATH, "models", "model.pkl")
    metrics_path = os.path.join(PROJECT_PATH, "metrics.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pkl not found at {model_path}")

    model = joblib.load(model_path)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    tracking_uri = f"file://{PROJECT_PATH}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Airflow_ML_Pipeline")

    with mlflow.start_run(run_name="airflow_ct_run"):
        # Логуємо метрики та модель
        mlflow.log_metrics({
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float))
        })
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

    # Реєструємо модель
    result = mlflow.register_model(
        model_uri=model_uri,
        name="SkinLesionClassifier",
    )

    # Переводимо у Staging
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    client.transition_model_version_stage(
        name="SkinLesionClassifier",
        version=result.version,
        stage="Staging",
    )

    print(
        f"[register_model] Model v{result.version} registered → Staging. "
        f"accuracy={metrics.get('accuracy', 'n/a'):.4f}"
    )


register_model = PythonOperator(
    task_id="register_model",
    python_callable=_register_model,
    dag=dag,
)

# ---------------------------------------------------------------------------
# Завдання 6b: Зупинка пайплайну (якість нижче порогу)
# ---------------------------------------------------------------------------
stop_pipeline = BashOperator(
    task_id="stop_pipeline",
    bash_command=(
        "echo '[stop_pipeline] Model accuracy is below threshold "
        f"({ACCURACY_THRESHOLD}). Registration skipped.'"
    ),
    dag=dag,
)

# ---------------------------------------------------------------------------
# Визначення залежностей (топологія DAG)
# ---------------------------------------------------------------------------
check_data >> prepare_data >> train_model >> evaluate_model >> check_quality
check_quality >> [register_model, stop_pipeline]
