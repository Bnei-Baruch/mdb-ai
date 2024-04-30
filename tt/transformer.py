from transformers import T5Tokenizer, T5ForConditionalGeneration

task_prefix = "translate English to Russian:"
model_checkpoint = "google/mt5-small"


# model_checkpoint = "Helsinki-NLP/fin-simple-mBART"


def transformer(txt):
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    input_ids = tokenizer(f"{task_prefix} {txt}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def transformer_2(sentence):
    model = T5ForConditionalGeneration.from_pretrained("gec-t5_small")
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    tokenized_sentence = tokenizer('gec: ' + sentence, max_length=128, truncation=True, padding='max_length',
                                   return_tensors='pt')
    corrected_sentence = tokenizer.decode(
        model.generate(
            input_ids=tokenized_sentence.input_ids,
            attention_mask=tokenized_sentence.attention_mask,
            max_length=128,
            num_beams=5,
            early_stopping=True,
        )[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return corrected_sentence
