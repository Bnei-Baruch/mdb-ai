import csv
import os
from urllib.request import urlopen


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
        with open("dataset.txt", 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['article', 'summary'])
            writer.writeheader()
            writer.writerows(self.ds)
        print("Saved dataset")


def fetch_file(uid):
    url = f"{os.getenv('LINKER_URL')}/{uid}"
    print(f"Fetching file {url}")
    return urlopen(url).read()
