# pip install datasets evaluate transformers[sentencepiece]
# pip install accelerate
# pip install rouge_score
# pip install nltk
import json

import nltk
import torch
from datasets import Dataset

#

torch.cuda.empty_cache()

with open('models/dataset.txt') as f:
    ds_data = f.read().replace("\\n", " ").replace("\\t", " ").replace("  ", " ")
    d = [x for x in json.loads(ds_data) if x["article"] is not None]
    ds = Dataset.from_list(d)
    ds.set_format(type="torch", columns=["article", "summary"])
    ds = ds.train_test_split(test_size=0.2)

# ds = load_dataset("billsum", split="ca_test")
# ds = ds.train_test_split(test_size=0.2)

from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModel, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, AutoModelForSeq2SeqLM

# model_checkpoint = "google/flan-t5-small"
#model_checkpoint = "google/mt5-small"
model_checkpoint = "google/mt5-base"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model

#
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
# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, quantization_config=q_config)
# model = prepare_model_for_kbit_training(model)
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, quantization_config=q_config)

# config = LoRAConfig(
#     r=8,
#     alpha=16,
#     intermediate_lora=True,
#     output_lora=True
# )

# model = AutoPeftModelForSeq2SeqLM(model, config=lora_config)
model = prepare_model_for_kbit_training(model)
adapter_name_he = "summ_he"
model.add_adapter(adapter_name=adapter_name_he, adapter_config=lora_config)
model = get_peft_model(model.base_model, lora_config)
# model.active_adapters = adapter_name_he

max_input_length = 4096
max_target_length = 200

prefix = "summarize: "


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
import evaluate

rouge_score = evaluate.load("rouge")

batch_size = 8
num_train_epochs = 8
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size

import numpy as np
from nltk.tokenize import sent_tokenize

nltk.download('punkt')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


# from transformers import DataCollatorForSeq2Seq
from transformers import DataCollatorForSeq2Seq

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

tokenized_datasets = tokenized_datasets.remove_columns(ds["train"].column_names)

features = [tokenized_datasets["train"][i] for i in range(2)]
data_collator(features)

# args = Seq2SeqTrainingArguments(
#     output_dir="summ_he",
#     auto_find_batch_size=True,
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     weight_decay=0.01,
#     save_total_limit=3,
#     num_train_epochs=4,
#     predict_with_generate=True,
#     fp16=True,
# )

# trainer = Seq2SeqTrainer(
#     model,
#     args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )

# small batch size to fit in memory
batch_size = 1

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

# for param in model.parameters():
#     param.requires_grad = False

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# model.train_adapter(adapter_name_he)
model.config.use_cache = False

print("start training...", model)
trainer.train()
print("start evaluating...")
#trainer.evaluate()
