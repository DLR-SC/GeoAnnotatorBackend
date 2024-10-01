from activeLearning.activeLearningUtils import compute_precision_recall_f1, calculate_A_at_k
from utils.baseModels import FeedbackRequest, Provider

import requests
import mlflow
import json
import os

'Feedback-Evaluation'
async def evaluateFeedback(feedback_data) -> None:
    precision, recall, f1_score, matched_coordinates = compute_precision_recall_f1(feedback_data, "predictions", "corrections")

    # MLFlow Tracking
    with mlflow.start_run(
        run_name="Feedback-Evaluation",
        tags={"feedback": "evaluation"},
        description="Evaluation of feedback → precision, recall, f1, accuracy for locations in ~ 10km and ~ 161km distances"
    ):
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1 Score", f1_score)
        mlflow.log_metric("A-161", round(calculate_A_at_k(matched_coordinates, 161),2))
        mlflow.log_metric("A-10", round(calculate_A_at_k(matched_coordinates, 10),2))

'Trigger retrain-job of model'
async def retrain_model(feedback_data, provider):
    response = requests.post(
        url='',
        json={
            "feedback_data": feedback_data,
            "provider": provider
        }, 
        headers={"Content-Type": "application/json"},
    )
    output=response.json()

    return output

'Retrain-Job-Check for Threshold'
async def check_feedback_threshold(provider: Provider, DIR_PATH) -> None:
    file_path = os.path.join(DIR_PATH, f"{provider.instance_name}_feedback.json")

    with open(file_path, "r") as f:
        feedback_data = json.load(f)
        if len(feedback_data) >= provider.data["threshold_retrain_job"]:
            # Evaluate feedback
            await evaluateFeedback(feedback_data)
            # Trigger retrain-job for model
            # await retrain_model(feedback_data, provider)
            print("Threshold achieved.")
            # Clear feedback-data
            # open(file_path, "w").close()

'Save data locally'
async def store_feedback(feedback: FeedbackRequest, DIR_PATH) -> None:
    file_path = os.path.join(DIR_PATH, f"{feedback.provider.instance_name}_feedback.json")

    try:
        with open(file_path, "r") as f:
            feedback_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        feedback_data = []

    'Restructure locations for further uses'
    async def restructure_locations(locations):
        return {location['name']: location['position'] for location in locations}

    feedback_data.append({
        "text": feedback.text,
        "predictions": await restructure_locations(feedback.predictions),
        "corrections": await restructure_locations(feedback.corrections)
    })

    with open(file_path, "w") as f:
        json.dump(feedback_data, f, indent=2)

'Return feedback data for specific provider'
async def load_feedback(instance_name: str, DIR_PATH) -> list[dict]:
    file_path = os.path.join(DIR_PATH, f"{instance_name}_feedback.json")

    try:
        with open(file_path, "r") as f:
            feedback_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        feedback_data = []

    return feedback_data