import os
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from transformers import MT5Tokenizer, AutoModelForSeq2SeqLM

model_checkpoint_fine_tuned = "./summarization/summ_he/checkpoint-1000"
model_checkpoint = "google/mt5-base"


def run_summarization(t):
    tokenizer = MT5Tokenizer.from_pretrained(model_checkpoint_fine_tuned, max_seq_len=4096)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint_fine_tuned)
    model_2 = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    prefix = "summarize:"
    input_text = f"{prefix} {t}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids, num_beams=4, no_repeat_ngram_size=2, min_length=100, max_length=300)
    print(tokenizer.decode(outputs[0]))
    print(tokenizer.decode(model_2.generate(input_ids)[0]))


if __name__ == '__main__':
    uid = '0li0gUJa'
    url = f"https://kabbalahmedia.info/assets/api/doc2text/{uid}"
    req = Request(url, headers={'Content-Type': 'text/html'})
    html = urlopen(req).read()
    txt = BeautifulSoup(html, 'html.parser')
    run_summarization(txt)
