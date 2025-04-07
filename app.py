import os
import signal
import logging
from time import sleep

from flask import Flask

from stt.handlers import handle_stt
from tt.handlers import handle_tt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

# Ignore SIGPIPE signal which is raised when client disconnects
signal.signal(signal.SIGPIPE, signal.SIG_IGN)

app = Flask(__name__)

@app.route('/stt', methods=['POST', 'GET'])
def stt():
    print("stt run")
    return handle_stt()


@app.route('/tt', methods=['POST', 'GET'])
def tt():
    return handle_tt()


if __name__ == '__main__':
    app.run()
