import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import csv
from collections import Counter
from tqdm.auto import tqdm, trange

tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
model = T5ForConditionalGeneration.from_pretrained('google/mt5-base')
df_ru = pd.read_csv('./heb_wikipedia_2021_10K/heb_wikipedia_2021_10K-sentences.txt', sep='\t', header=None,
                    quoting=csv.QUOTE_NONE)
df_ru.columns = ['idx', 'text']
cnt_ru = Counter()
for text in tqdm(df_ru.text):
    cnt_ru.update(tokenizer.encode(text))
print(len(cnt_ru), len(cnt_ru) / tokenizer.vocab_size)

new_tokens = set(range(1000))
# for i, (k, v) in enumerate(cnt_en.most_common(10_000)):
#     if k not in new_tokens:
#         new_tokens.add(k)
for i, (k, v) in enumerate(cnt_ru.most_common(25_000)):
    if len(new_tokens) == 29_900:
        print(i, 'Russan tokens are included')
        break
    if k not in new_tokens:
        new_tokens.add(k)
for t in range(tokenizer.vocab_size - 100, tokenizer.vocab_size):
    new_tokens.add(t)
print(len(new_tokens))
kept_ids = sorted(new_tokens)

new_size = len(kept_ids)
new_emb = torch.nn.Embedding(new_size, model.shared.embedding_dim)
new_head = torch.nn.Linear(in_features=model.lm_head.in_features, out_features=new_size, bias=False)
for new_id, old_id in enumerate(kept_ids):
    new_emb.weight.data[new_id] = model.shared.weight.data[old_id]
    new_head.weight.data[new_id] = model.lm_head.weight.data[old_id]
model.shared.weight = new_emb.weight
model.lm_head.weight = new_head.weight
model.config.__dict__['vocab_size'] = new_size
model.config.__dict__['_name_or_path'] = 'cointegrated/he_t5_cut'
model.save_pretrained('he_t5_cut')

# ! wget https://raw.githubusercontent.com/google/sentencepiece/master/src/sentencepiece_model.proto
# ! protoc --python_out=. sentencepiece_model.proto
import sentencepiece.sentencepiece_model_pb2 as spmp

smp = tokenizer.sp_model.serialized_model_proto()
m = spmp.ModelProto()
m.ParseFromString(smp)
print('the loaded model has pieces:', len(m.pieces))
new_pieces = [m.pieces[idx] for idx in kept_ids]
print('the new pieces:', len(new_pieces))
# replace the content of the first 30K pieces
for i, p in enumerate(new_pieces):
    m.pieces[i].piece = p.piece
    m.pieces[i].score = p.score
    m.pieces[i].type = p.type
# drop the remaining pieces
n = len(new_pieces)
for i in trange(len(m.pieces) - n):
    m.pieces.pop(len(m.pieces) - 1)
print(len(m.pieces))
with open('new_sp.model', 'wb') as f:
    f.write(m.SerializeToString())
new_tokenizer = T5Tokenizer('new_sp.model', extra_ids=0)
new_tokenizer.save_pretrained('he_t5_cut')
