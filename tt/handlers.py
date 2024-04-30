import os
import tempfile

from flask import request, Response, send_file

from const import RESULT_PATH_DOCX
from tt.srt_to_docx import srt_to_paragraphs, save_to_docx


def handle_tt():
    lang = request.values['lang']
    if 'txt' not in request.files:
        return Response('must send or file', status=422)

    srt = request.files['txt']
    with tempfile.TemporaryDirectory() as temp_dir:
        path = tt_from_file(srt, lang, temp_dir)
    return send_file(path, download_name=f"{srt.filename.split('.')[0]}.docx")


def tt_from_file(srt, lang, tmp_dir):
    print("from_url build subs")
    srt_path = os.path.join(tmp_dir, srt.filename)
    srt.save(srt_path)
    ps = srt_to_paragraphs(srt_path, lang)
    path = os.path.abspath(RESULT_PATH_DOCX)
    # path = os.path.join(tmp_dir, f"{srt.filename.split('.')[0]}.docx")
    save_to_docx(ps, lang, path)
    return path
