import os
from urllib.request import urlretrieve

from flask import request, send_file, Response, jsonify

from const import AUDIO_PATH
from stt.to_subs import ToSubs
from stt.transcriber import transcriber


def handle_stt():
    if 'lang' not in request.values:
        return Response('language is must', status=422)
    lang = request.values['lang']
    if 'url' not in request.values and len(request.files) == 0:
        return Response('must send or file or url', status=422)
    if len(request.files) > 0:
        res = stt_from_file(list(request.files)[0], lang)
    else:
        res = stt_from_url(request.values['url'], lang)
    if 'tr_only' in request.values:
        return jsonify(res)

    path = os.path.abspath('result.srt')
    res.save(path)
    print("result file ready", path)
    return send_file(path)


def stt_from_file(f, lang):
    f.save(AUDIO_PATH)
    word_timestamp = transcriber(lang)
    return ToSubs(word_timestamp, lang).run()


def stt_from_url(url, lang):
    print("from_url", url, lang)
    urlretrieve(url, AUDIO_PATH)
    word_timestamp = transcriber(lang)
    # word_timestamp = transcriber_on_side(url, lang)
    if 'tr_only' in request.values:
        return word_timestamp

    print("from_url build subs")
    return ToSubs(word_timestamp, lang).run()
