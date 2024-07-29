import os
from time import sleep

from flask import Flask

from stt.handlers import handle_stt
from summarization.handlers import handle_summary
from tt.handlers import handle_tt

app = Flask(__name__)

WAIT_PATH = "wait.txt"


def wait_ready():
    if os.path.exists(WAIT_PATH):
        sleep(1000)
        wait_ready()


@app.before_request
def before_request():
    wait_ready()
    with open(WAIT_PATH, 'w') as fp:
        pass


@app.after_request
def add_header(response):
    os.remove(WAIT_PATH)
    return response


@app.route('/stt', methods=['POST'])
def stt():
    print("stt run")
    return handle_stt()


@app.route('/tt', methods=['POST'])
def tt():
    return handle_tt()


@app.route('/sum', methods=['POST', 'GET'])
def summary():
    return handle_summary()


try:
    os.remove(WAIT_PATH)
except OSError:
    pass

if __name__ == '__main__':
    app.run()
