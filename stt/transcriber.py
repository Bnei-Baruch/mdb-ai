import os

import requests
from faster_whisper import WhisperModel
from flask import json

from const import AUDIO_PATH, MODEL_BY_LANG


def transcriber(lang):
    print("transcriber start model")
    m = WhisperModel(MODEL_BY_LANG[lang], device="cuda", compute_type="float16")
    segments, info = m.transcribe(AUDIO_PATH, language=lang, beam_size=5, best_of=5, vad_filter=True,
                                  condition_on_previous_text=False, word_timestamps=True)

    print("transcriber end model")
    results = {}
    idx = 0
    print("segments to object start")
    for segment in segments:
        for word in segment.words:
            results[idx] = {'start': word.start, 'end': word.end, 'word': word.word}
            idx += 1
    print("segments to object end")
    return results


def transcriber_on_side(url, lang):
    mock_path = r"/Users/davgur/PycharmProjects/mdb-ai/mock.json"
    if os.path.exists(mock_path):
        with open(mock_path, 'r') as f:
            return json.loads(f.read())
    params = {'lang': lang, 'url': url, 'tr_only': True}
    r = requests.get("http://10.65.4.100:5000/stt", params=params)
    j = json.loads(r.content)
    with open(mock_path, 'w') as out_f:
        json.dump(j, out_f)
    return j
