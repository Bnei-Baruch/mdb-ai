from flask import make_response, request

from summarization.helper import fetch_file
from summarization.use_model import run_summarization


def handle_summary():
    uid = request.values['uid']
    txt = fetch_file(uid)
    try:
        rez = run_summarization(txt)
    except Exception as e:
        return make_response(str(e), 500)
    return make_response(rez, 200)
