from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from transformers import MT5Tokenizer, AutoModelForSeq2SeqLM

# model_checkpoint_fine_tuned = "./summarization/summ_he_conditional/checkpoint-1000"
model_checkpoint_fine_tuned = "./summarization/summ_he_large/checkpoint-1000"


def run_summarization(t):
    tokenizer = MT5Tokenizer.from_pretrained(model_checkpoint_fine_tuned, max_seq_len=4096)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint_fine_tuned)

    prefix = "summarize:"
    input_text = f"{prefix} {t}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids, num_beams=4, no_repeat_ngram_size=2, min_length=100, max_length=300)
    res = tokenizer.decode(outputs[0])
    print(res)
    return res


if __name__ == '__main__':
    uid = '0li0gUJa'
    url = f"https://kabbalahmedia.info/assets/api/doc2text/{uid}"
    req = Request(url, headers={'Content-Type': 'text/html'})
    html = urlopen(req).read()
    txt = BeautifulSoup(html, 'html.parser')
    run_summarization(txt)
