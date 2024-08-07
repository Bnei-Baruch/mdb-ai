import json

from datasets import Dataset


def prepare_dataset():
    with open('models/dataset.txt') as f:
        ds_data = f.read().replace("\\n", " ").replace("\\t", " ").replace("  ", " ")
        d = [x for x in json.loads(ds_data) if x["article"] is not None]
        ds = Dataset.from_list(d)
        ds.set_format(type="torch", columns=["article", "summary"])
        ds = ds.train_test_split(test_size=0.2)
    return ds
