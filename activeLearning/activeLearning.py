from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

# from celery import Celery
# import redis
import subprocess
import json

# celery_app = Celery('tasks', broker='redis://localhost:6379/0')

model_name="Meta-Llama-3.1-8B-Instruct"

# def is_redis_available():
#   try:
#     r = redis.StrictRedis(host='localhost', port=6379, db=0)
#     r.ping()
#     print("Server running")
#   except redis.ConnectionError:
#     r = redis.Redis(
#       host='localhost',
#       port=6379,
#       db=0,
#       decode_responses=True,
#       retry_on_timeout=True
#     )
#     response = r.ping()
#     print(f"Server started.")

# @celery_app.task
def retrain_model():
  model, tokenizer = FastLanguageModel.from_pretrained(
      # model_name = f"models/{model_name}",
      model_name = f"unsloth/{model_name}",
      dtype = None,
      load_in_4bit = True,
      max_seq_length = 2048,
  )

  # model = FastLanguageModel.get_peft_model(
  #   model,
  #   r = 16,
  #   target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
  #                     "gate_proj", "up_proj", "down_proj",],
  #   lora_alpha = 16,
  #   lora_dropout = 0,
  #   bias = "none",
  #   use_gradient_checkpointing = "unsloth",
  #   random_state = 3407,
  #   use_rslora = False,
  #   loftq_config = None,
  # )

  # alpaca_prompt = """You are an assitant that strictly extracts geographic references from the input. For each location, provide the place-name (exactly as in the text), the latitude and the longitude of the place as a json-object, like {{ name: place-name, position: [latitude, longitude] }}. Create a json-list out of these objects. In the list, there should be no repetitive places with the same place-name. Please only return the value with no explanation or further information and as a normal text without labeling it as json.

  # ### Input:
  # {}

  # ### Output:
  # {}"""

  # EOS_TOKEN = tokenizer.eos_token
  # def formatting_prompts_func(examples):
  #     inputs       = examples["text"]
  #     outputs      = examples["corrections"]
  #     texts = []
  #     for input, output in zip(inputs, outputs):
  #       output_str = json.dumps(output, indent=2)
  #       text = alpaca_prompt.format(input, output_str) + EOS_TOKEN
  #       texts.append(text)
  #     return { "text" : texts, }
  # pass

  # dataset = load_dataset("json", data_files="../feedback/test.json", split='train')
  # dataset = dataset.map(formatting_prompts_func, batched = True,)

  # # Counting data for training steps
  # def Datacount():
  #     try:
  #         with open("feedback/test.json", 'r') as f:
  #             data = json.load(f)
  #         return len(data)
  #     except:
  #         return 60

  # trainer = SFTTrainer(
  #   model = model,
  #   tokenizer = tokenizer,
  #   train_dataset = dataset,
  #   dataset_text_field = "text",
  #   max_seq_length = 2048,
  #   dataset_num_proc = 2,
  #   packing = False,
  #   args = TrainingArguments(
  #     per_device_train_batch_size = 2,
  #     gradient_accumulation_steps = 4,
  #     warmup_steps = 5,
  #     # num_train_epochs = 2,
  #     max_steps = Datacount(),
  #     learning_rate = 2e-4,
  #     fp16 = not is_bfloat16_supported(),
  #     bf16 = is_bfloat16_supported(),
  #     logging_steps = 1,
  #     optim = "adamw_8bit",
  #     weight_decay = 0.01,
  #     lr_scheduler_type = "linear",
  #     seed = 3407,
  #     output_dir = "outputs",
  #   ),
  # )

  # trainer_stats = trainer.train()

  # Modell speichern
  model.save_pretrained(f"models/{model_name}") # Local saving
  tokenizer.save_pretrained(f"models/{model_name}")

  if True: model.save_pretrained_gguf(f"models/{model_name}/gguf", tokenizer, quantization_method = "q4_k_m")
  print("Modell wurde neu trainiert und gespeichert.")

retrain_model()
# is_redis_available()
  # MLFlow Tracking
  # with mlflow.start_run(
  #     run_name="Retrain-Job",
  #     tags={"job": "retrain"},
  #     description="Retrain-Job of configured model in provider"
  # ):
      # Speichern der Metriken in MLFlow
      # mlflow.log_metric("precision", eval_results['eval_precision'])
      # mlflow.log_metric("recall", eval_results['eval_recall'])
      # mlflow.log_metric("f1", eval_results['eval_f1'])

      # Speichern der Trainingsparameter
      # mlflow.log_param("learning_rate", training_args.learning_rate)
      # mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
      # mlflow.log_param("num_train_epochs", training_args.num_train_epochs)