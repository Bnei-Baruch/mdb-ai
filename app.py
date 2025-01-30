import os
from time import sleep

from flask import Flask

from stt.handlers import handle_stt
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


@app.route('/stt', methods=['POST', 'GET'])
def stt():
    print("stt run")
    return handle_stt()


@app.route('/tt', methods=['POST', 'GET'])
def tt():
    return handle_tt()


try:
    os.remove(WAIT_PATH)
except OSError:
    pass

if __name__ == '__main__':
    app.run()
