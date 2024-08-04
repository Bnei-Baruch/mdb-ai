# pip install datasets evaluate transformers[sentencepiece]
# pip install accelerate
# pip install rouge_score
# pip install nltk
import json

import nltk
import torch

from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from transformers.utils.quantization_config import QuantizationMethod

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

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoModelForCausalLM

model_checkpoint = "google/flan-t5-small"
# model_checkpoint = "./models/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, load_in_8bit=True, device_map="auto")

from peft import LoraConfig, get_peft_model, TaskType

l_config = LoraConfig(
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
    bnb_4bit_compute_dtype=torch.bfloat16,
    quant_method=QuantizationMethod.BITS_AND_BYTES,
    lora_config=l_config
)

# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, load_in_8bit=True, device_map="auto")
# model = prepare_model_for_int8_training(model)
# model = get_peft_model(model, config)
# model.print_trainable_parameters()
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_checkpoint,
    load_in_8bit=True,
    device_map="auto"
)
model = get_peft_model(model, l_config)

from transformers import Seq2SeqTrainingArguments

max_input_length = 2048
max_target_length = 50

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
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

tokenized_datasets = tokenized_datasets.remove_columns(ds["train"].column_names)

features = [tokenized_datasets["train"][i] for i in range(2)]
data_collator(features)

from transformers import Seq2SeqTrainer

args = Seq2SeqTrainingArguments(
    output_dir="summ_he",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.evaluate()
