import csv
from dotenv import load_dotenv

from summarization.dataset import fetch_mdb
from summarization.dataset.fetch_text import TextDataset

load_dotenv()


def run():
    mdb_d = fetch_mdb.run()
    ds = TextDataset()
    ds.run(mdb_d)


run()
