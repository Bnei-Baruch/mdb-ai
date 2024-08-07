"""
Run the ORPO training script with the following command with some example arguments.
In general, the optimal configuration for ORPO will be similar to that of DPO without the need for a reference model:

# regular:
python examples/scripts/orpo.py \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 8e-6 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="gpt2-aligned-orpo" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns

# peft:
python examples/scripts/orpo.py \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 8e-5 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="gpt2-lora-aligned-orpo" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
"""
import multiprocessing

from datasets import load_dataset
from peft import LoraConfig, TaskType
from trl import ORPOTrainer, get_peft_config, AutoModelForSeq2SeqLMWithValueHead

from summarization.helper import prepare_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ds = prepare_dataset()
model_checkpoint = "google/mt5-small"
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_checkpoint, load_in_4bit=True, peft_config=lora_config)


def process(row):
    row["prompt"] = tokenizer.apply_chat_template(row["chosen"][:-1], tokenize=False)
    row["chosen"] = tokenizer.apply_chat_template([row["chosen"][-1]], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template([row["rejected"][-1]], tokenize=False)
    return row


ds = ds.map(
    process,
    num_proc=1 if orpo_args.debug else multiprocessing.cpu_count(),
    load_from_cache_file=False,
)
train_dataset = ds["train"]
eval_dataset = ds["test"]

################
# Training
################
trainer = ORPOTrainer(
    model,
    args=orpo_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    peft_config=get_peft_config(model_config),
)

# train and save the model
trainer.train()
trainer.save_model(orpo_args.output_dir)
