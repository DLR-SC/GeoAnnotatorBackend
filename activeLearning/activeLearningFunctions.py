from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
# import mlflow
# import torch

from math import radians, sin, cos, sqrt, atan2
from utils.baseModels import FeedbackRequest, Provider
import json
import os


def calculate_distance(coord1, coord2):
    # Function to calculate distance between two coordinates
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6371 * c  # Radius of Earth in kilometers
    return distance

def compute_precision_recall_f1(instances, _truth, _pred):
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    matched_coordinates = []

    for instance in instances:
        ground_truth = instance[_truth]
        predicted = instance[_pred]
        matched_ground_truth = set()  # To keep track of matched ground truth elements
        
        true_positives = 0
        for pred_key, pred_coord in predicted.items():
            matched = False
            for gt_key, gt_coord in ground_truth.items():
                if pred_key.lower() in gt_key.lower() or gt_key.lower() in pred_key.lower():
                    if gt_key not in matched_ground_truth:  # Ensure we don't count the same ground truth multiple times
                        true_positives += 1
                        matched_ground_truth.add(gt_key)
                        matched_coordinates.append((pred_coord, gt_coord))
                        matched = True
                        break
            
            # False positives are elements in predicted that did not match any ground truth element
            if not matched:
                total_false_positives += 1
        
        # False negatives are ground truth elements that did not match any predicted element
        total_false_negatives += len(ground_truth) - len(matched_ground_truth)
        total_true_positives += true_positives
    
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score, matched_coordinates

def calculate_A_at_k(matched_coordinates, k):
    # Function to calculate accuracy at k (A@k)
    correct_matches = 0
    for pred_coord, truth_coord in matched_coordinates:
        if calculate_distance(pred_coord, truth_coord) <= k:
            correct_matches += 1

    accuracy_at_k = (correct_matches / len(matched_coordinates)) * 100 if matched_coordinates else 0
    return accuracy_at_k

'Funktion zum Retraining des Modells mit neuen Daten'
def retrain_model(feedback_data, provider):
   # Preprocessing for Finetuning
    texts = []
    labels = []

    for entry in feedback_data:
        text = entry["text"]
        corrected_labels = entry["corrections"]
        
        texts.append(text)
        labels.append(corrected_labels)

    # Hugging Face Dataset
    train_dataset = Dataset.from_dict({"text": texts, "labels": labels})

    # Lade das LLaMA-Modell und den Tokenizer
    model_name = "path/to/llama-3.1-8b"  # Gib den Pfad zum GGUF-Modell an
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,  # Für 8-Bit-Training (speicher- und performance-effizient)
        device_map="auto"
    )

    # Konfiguriere QLoRA für effizientes Feintuning
    lora_config = LoraConfig(
        r=4,                       
        lora_alpha=16,             
        lora_dropout=0.05,         
        target_modules=["q_proj", "v_proj"],  
        bias="none",               
        task_type="CAUSAL_LM"      
    )

    # PEFT-Modell erstellen
    peft_model = get_peft_model(model, lora_config)

    # Trainingsargumente konfigurieren
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        save_steps=500,
        save_total_limit=2
    )

    # Trainer initialisieren
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Fine-Tuning durchführen
    trainer.train()

    # Evaluation durchführen
    # eval_results = trainer.evaluate()

    # Modell speichern
    trainer.save_model("./fine_tuned_model")
    print("Modell wurde neu trainiert und gespeichert.")

    #  # MLFlow Tracking starten
    # mlflow.start_run()

    # # Speichern der Metriken in MLFlow
    # mlflow.log_metric("precision", eval_results['eval_precision'])
    # mlflow.log_metric("recall", eval_results['eval_recall'])
    # mlflow.log_metric("f1", eval_results['eval_f1'])

    # # Speichern der Trainingsparameter
    # mlflow.log_param("learning_rate", training_args.learning_rate)
    # mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
    # mlflow.log_param("num_train_epochs", training_args.num_train_epochs)

    # mlflow.end_run()

'Retrain-Job-Check for Threshold'
async def check_feedback_threshold(provider: Provider, DIR_PATH) -> None:
    file_path = os.path.join(DIR_PATH, f"{provider.instance_name}_feedback.json")

    with open(file_path, "r") as f:
        feedback_data = json.load(f)
        if len(feedback_data) >= provider.data["threshold_retrain_job"]: 
            # retrain_model(feedback_data, provider)
            print("Threshold achieved.")
            # Nach dem Training wird die Datei geleert
            open(file_path, "w").close()

# Restructure locations for further uses
async def restructure_locations(locations):
    restructured = {location['name']: location['position'] for location in locations}
    print(restructured)
    return restructured

'Save data locally'
async def store_feedback(feedback: FeedbackRequest, DIR_PATH):
    file_path = os.path.join(DIR_PATH, f"{feedback.provider.instance_name}_feedback.json")

    try:
        with open(file_path, "r") as f:
            feedback_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        feedback_data = []

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