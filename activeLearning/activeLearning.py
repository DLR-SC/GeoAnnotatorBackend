from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset

'Funktion zum Retraining des Modells mit neuen Daten'
async def retrain_model(feedback_data, provider):
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