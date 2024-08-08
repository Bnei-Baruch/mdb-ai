import json

import evaluate
import nltk
import numpy as np
import torch
from datasets import Dataset
from nltk.tokenize import sent_tokenize
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
from transformers import MT5Tokenizer, BitsAndBytesConfig, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

torch.cuda.empty_cache()

with open('models/dataset.txt') as f:
    ds_data = f.read().replace("\\n", " ").replace("\\t", " ").replace("  ", " ")
    d = [x for x in json.loads(ds_data) if x["article"] is not None]
    ds = Dataset.from_list(d)
    ds.set_format(type="torch", columns=["article", "summary"])
    ds = ds.train_test_split(test_size=0.2)

# model_checkpoint = "google/mt5-small"
model_checkpoint = "google/mt5-base"

tokenizer = MT5Tokenizer.from_pretrained(model_checkpoint)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,

)
q_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

max_input_length = 4096
max_target_length = 200
prefix = "summarize: "
label_pad_token_id = -100
# small batch size to fit in memory
batch_size = 1
num_train_epochs = 8

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, quantization_config=q_config)
model = prepare_model_for_kbit_training(model)
adapter_name_he = "summ_he"
model.add_adapter(adapter_name=adapter_name_he, adapter_config=lora_config)
model = get_peft_model(model.base_model, lora_config)


def preprocess_function(examples):
    i = 0
    for ex in examples["article"]:
        if ex is None:
            print(f"article not found {examples['summary'][i]} , {i}")
        i += 1

    inputs = [prefix + txt for txt in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = ds.map(preprocess_function, batched=True)
print("tokenize of dataset was ended")

rouge_score = evaluate.load("rouge")
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size
nltk.download('punkt')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge_score.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    print(list(result.keys()))
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

tokenized_datasets = tokenized_datasets.remove_columns(ds["train"].column_names)

features = [tokenized_datasets["train"][i] for i in range(2)]
data_collator(features)

training_args = Seq2SeqTrainingArguments(
    learning_rate=3e-4,
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_steps=200,
    output_dir="./summ_he",
    overwrite_output_dir=True,
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

model.config.use_cache = False
model.config.max_position_embeddings = max_input_length

print("start training")
trainer.train()
print("end of training", model)
