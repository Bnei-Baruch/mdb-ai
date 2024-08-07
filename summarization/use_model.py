text = """
העקרון הדתי, מתוך שלא לשמה בא לשמה והכין ההשגחה הנהגת הבריות בדרך אגואיסטית שבהכרח תביאה לחורבן העולם אם לא יקבלו הדת להשפיע. וע"כ יש בה צורך פרגמטי. ומתוכה בא לשמה. מהו צורך נפשי כמו שלעור לא יושג צבעים ולסריס אהבת המין כן החסר הצורך נפשי א"א לצייר לו הצורך הזה. אמנם צורך מחויב הוא.
"""

from transformers import MT5Tokenizer, AutoModelForSeq2SeqLM

model_checkpoint_fine_tuned = "./summ_he/checkpoint-1000"
model_checkpoint = "google/mt5-base"


def run_summarization(t):
    tokenizer = MT5Tokenizer.from_pretrained(model_checkpoint_fine_tuned, max_seq_len=4096)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint_fine_tuned)
    model_2 = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    prefix = "summarize:"
    input_text = f"{prefix} {t}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0]))
    print(tokenizer.decode(model_2.generate(input_ids)[0]))


run_summarization(text)
