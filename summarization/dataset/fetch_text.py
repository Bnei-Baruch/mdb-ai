import csv
import json
import os
from urllib.request import urlopen, Request

from bs4 import BeautifulSoup


class TextDataset:
    def __init__(self):
        self.ds = []

    def run(self, mdb_data):
        i = 0
        for [a_uid, s_uid] in mdb_data:
            i += 1
            print(f"Processing, iteration num {i}")
            try:
                article = fetch_file(a_uid)
                summary = fetch_file(s_uid)
            except Exception as e:
                print(f"File not fetched:{e}")
                continue
            self.ds.append({"article": article, "summary": summary})
        self.save_dataset()

    def save_dataset(self):
        with open("dataset.txt", 'w', encoding='utf8') as f:
            f.write(json.dumps(self.ds, ensure_ascii=False))
        print("Saved dataset")


def fetch_file(uid):
    url = f"{os.getenv('LINKER_URL')}/{uid}"
    req = Request(url, headers={'Content-Type': 'text/html'})
    html = urlopen(req).read()
    txt = BeautifulSoup(html, 'html.parser')
    return txt.string
