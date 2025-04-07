import os
from urllib.request import urlretrieve
import tempfile

from flask import request, send_file, Response, jsonify, after_this_request
import logging

from const import AUDIO_PATH
from stt.to_subs import ToSubs
from stt.transcriber import transcriber


def handle_stt():
    try:
        if 'lang' not in request.values:
            return Response('language is must', status=422)
        lang = request.values['lang']
        if 'url' not in request.values and len(request.files) == 0:
            return Response('must send or file or url', status=422)
        if len(request.files) > 0:
            res = stt_from_file(request.files.get("file"), lang)
        else:
            res = stt_from_url(request.values['url'], lang)
        if 'tr_only' in request.values:
            return jsonify(res)

        # Create a temporary file instead of using a fixed path
        temp_file = tempfile.NamedTemporaryFile(suffix='.srt', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        res.save(temp_path)
        print("result file ready", temp_path)
        
        @after_this_request
        def cleanup(response):
            # Delete the file after the response has been sent
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    print(f"Temporary file {temp_path} deleted")
            except Exception as e:
                logging.error(f"Failed to delete temporary file: {str(e)}")
            return response
        
        return send_file(temp_path, as_attachment=True, download_name='subtitles.srt')
    except BrokenPipeError:
        # Client disconnected before response was sent
        logging.warning(f"Client disconnected during STT process: {request.values.get('url', 'N/A')}")
        return '', 499  # Return a status code for client closed request
    except OSError as e:
        if 'Broken pipe' in str(e):
            logging.warning(f"Broken pipe during STT process: {request.values.get('url', 'N/A')}")
            return '', 499
        # Re-raise other OSErrors
        logging.error(f"OSError during STT process: {str(e)}")
        raise


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
