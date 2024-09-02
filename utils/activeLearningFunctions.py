from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline, AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import mlflow
import torch
import json

def compute_metrics(pred):
    'Berechnung der Metriken Precision, Recall und F1'
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

'Funktion zum Retraining des Modells mit neuen Daten'
def retrain_model(feedback_data, _model_):
    match _model_:
        case "BERT":
            model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
            model = pipeline("ner", model=model_name, tokenizer="bert-base-cased")
        case "GPT2":
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        case _:
            model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
            model = pipeline("ner", model=model_name, tokenizer="bert-base-cased")

    # Vorbereitung der Daten für das Fine-Tuning
    texts = []
    labels = []
    for entry in feedback_data:
        texts.append(entry["text"])
        # Dummy-Label-Erstellung für Beispiel; hier würde das korrekte Labeling erfolgen
        labels.append([1] * len(tokenizer(entry["text"])["input_ids"])) 

    # Datensatz erstellen
    dataset = Dataset.from_dict({"text": texts, "labels": labels})
    dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'), batched=True)

    # Trainingsparameter
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        # save_steps=10_000,
        # save_total_limit=2,
        fp16=True,  # Mixed precision training für effizienteres Training
    )

    # Trainer initialisieren
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    # Fine-Tuning durchführen
    trainer.train()

     # Evaluation durchführen
    eval_results = trainer.evaluate()

    # Modell speichern
    trainer.save_model("./fine_tuned_model")
    print("Modell wurde neu trainiert und gespeichert.")

     # MLFlow Tracking starten
    mlflow.start_run()

    # Speichern der Metriken in MLFlow
    mlflow.log_metric("precision", eval_results['eval_precision'])
    mlflow.log_metric("recall", eval_results['eval_recall'])
    mlflow.log_metric("f1", eval_results['eval_f1'])

    # Speichern der Trainingsparameter
    mlflow.log_param("learning_rate", training_args.learning_rate)
    mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
    mlflow.log_param("num_train_epochs", training_args.num_train_epochs)

    mlflow.end_run()

'Überprüfung der Anzahl der Feedback-Datensätze'
def check_feedback_threshold():
    with open("feedback.json", "r") as f:
        feedback_lines = [json.loads(line) for line in f.readlines()]
        if len(feedback_lines) >= 100: 
            retrain_model(feedback_lines)
            # Nach dem Training wird die Datei geleert
            open("feedback.json", "w").close()

'Funktion zum Speichern von Feedback'
def store_feedback(original_text, prediction, corrected_coordinates):
    feedback_entry = {
        "text": original_text,
        "prediction": prediction,
        "correction": corrected_coordinates
    }
    with open("feedback.json", "a") as f:
        json.dump(feedback_entry, f)
        f.write("\n")

'Die Feedbacks vom Frontend verarbeiten, also speichern und prüfen, ob die Grenze mit dem Schwellenwert 100 erreicht wurde'
def process_feedback(text, prediction, user_correction):
    # Speichern des Feedbacks
    store_feedback(text, prediction, user_correction)
    # Überprüfen, ob Schwelle erreicht ist
    check_feedback_threshold()