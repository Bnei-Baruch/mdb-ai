from transformers import pipeline

text = """
חלק ג', קטע ב', כותרת "העקרון הדתי, מתוך שלא לשמה בא לשמה"

קריין: ספר "כתבי בעל הסולם", עמ' 848, "כתבי הדור האחרון", חלק ג', קטע ב', כותרת: "העקרון הדתי, מתוך שלא לשמה בא לשמה".

העקרון הדתי, מתוך שלא לשמה בא לשמה
"והכין ההשגחה הנהגת הבריות בדרך אגואיסטית שבהכרח תביאה לחורבן העולם אם לא יקבלו הדת להשפיע. וע"כ יש בה צורך פרגמטי. ומתוכה בא לשמה."

מהו צורך נפשי
"כמו שלעור לא יושג צבעים ולסריס אהבת המין כן החסר הצורך נפשי א"א לצייר לו הצורך הזה. אמנם צורך מחויב הוא."
"""


def run_summarization(t):
    summarizer = pipeline("summarization", model="./summ_he/checkpoint-268")
    summ = summarizer(f"summarize: {t}")
    print(summ)


run_summarization(text)
