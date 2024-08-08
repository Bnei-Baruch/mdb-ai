import json
import os
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import psycopg2
from bs4 import BeautifulSoup
from datasets import Dataset


def connect_mdb():
    mdb_str = os.getenv("MDB")
    url = urlparse(mdb_str)
    return psycopg2.connect(dbname=url.path[1:], user=url.username,
                            password=url.password, host=url.hostname)


def prepare_dataset():
    with open('models/dataset.txt') as f:
        ds_data = f.read().replace("\\n", " ").replace("\\t", " ").replace("  ", " ")
        d = [x for x in json.loads(ds_data) if x["article"] is not None]
        ds = Dataset.from_list(d)
        ds.set_format(type="torch", columns=["article", "summary"])
        ds = ds.train_test_split(test_size=0.2)
    return ds


def fetch_file(uid):
    url = f"{os.getenv('LINKER_URL')}/{uid}"
    req = Request(url, headers={'Content-Type': 'text/html'})
    html = urlopen(req).read()
    txt = BeautifulSoup(html, 'html.parser')
    return txt.string


def clear_txt(txt):
    return txt.replace("\\n", " ").replace("\\t", " ").replace("  ", " ")
