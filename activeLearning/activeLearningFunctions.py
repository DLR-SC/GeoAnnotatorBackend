from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
# import mlflow
# import torch

from utils.baseModels import FeedbackRequest, Provider
import json
import os

# def compute_metrics(pred):
#     'Berechnung der Metriken Precision, Recall und F1'
#     labels = pred.label_ids, preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
#     return {
#         'precision': precision,
#         'recall': recall,
#         'f1': f1
#     }

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

'Save data locally'
async def store_feedback(feedback: FeedbackRequest, DIR_PATH):
    file_path = os.path.join(DIR_PATH, f"{feedback.provider.instance_name}_feedback.json")

    try:
        with open(file_path, "r") as f:
            feedback_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        feedback_data = []

    feedback_entry = {
        "text": feedback.text,
        "predictions": feedback.predictions,
        "corrections": feedback.corrections
    }
    feedback_data.append(feedback_entry)

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