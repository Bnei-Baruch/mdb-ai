from transformers import T5Tokenizer, T5ForConditionalGeneration

# model_name = "google/mt5-base"
model_name = "google/umt5-small"
# model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


def run(txt, lang="he"):
    txt_list = txt.split()
    # input_text = "Статья: " + " ".join(txt_list)
    input_text = "summarize: " + " ".join(txt_list)
    # res = ["מאמר: ", txt.split()]
    # input_text = "Article:" + ''.join(txt.splitlines())
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
