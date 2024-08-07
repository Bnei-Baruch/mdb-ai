import json

model_id = "google/flan-t5-small"
# model_id="google/flan-t5-xl"
# model_id="google/flan-t5-xxl"
peft_model_id = "results-ft-model"
data_train = "data/train"
logs_output_dir = "logs"

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(model_id)

from datasets import load_from_disk, Dataset
from random import randrange


def prepare_dataset():
    with open('models/dataset.txt') as f:
        ds_data = f.read().replace("\\n", " ").replace("\\t", " ").replace("  ", " ")
        d = [x for x in json.loads(ds_data) if x["article"] is not None]
        ds = Dataset.from_list(d)
        ds.set_format(type="torch", columns=["article", "summary"])
        ds = ds.train_test_split(test_size=0.2)
    return ds


# ds = load_from_disk(data_train)
ds = prepare_dataset()

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

from transformers import DataCollatorForSeq2Seq

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=logs_output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3,
    num_train_epochs=5,
    logging_dir=f"{logs_output_dir}/logs",
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="no",
    report_to="tensorboard",
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=ds,
)
model.config.use_cache = False

trainer.train()

trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
