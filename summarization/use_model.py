from transformers import pipeline


text = """
חלק ג', קטע ב', כותרת "העקרון הדתי, מתוך שלא לשמה בא לשמה"

קריין: ספר "כתבי בעל הסולם", עמ' 848, "כתבי הדור האחרון", חלק ג', קטע ב', כותרת: "העקרון הדתי, מתוך שלא לשמה בא לשמה".

העקרון הדתי, מתוך שלא לשמה בא לשמה
"והכין ההשגחה הנהגת הבריות בדרך אגואיסטית שבהכרח תביאה לחורבן העולם אם לא יקבלו הדת להשפיע. וע"כ יש בה צורך פרגמטי. ומתוכה בא לשמה."

מהו צורך נפשי
"כמו שלעור לא יושג צבעים ולסריס אהבת המין כן החסר הצורך נפשי א"א לצייר לו הצורך הזה. אמנם צורך מחויב הוא."
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration, MT5ForConditionalGeneration

model_checkpoint_fine_tuned = "./summ_he/checkpoint-536"
def run_summarization(t):
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint_fine_tuned)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint_fine_tuned)
    model_base = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    print(model_base.parameters())

    prefix = "summarize:"
    input_text = f"{prefix} t"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0]))


run_summarization(text)
