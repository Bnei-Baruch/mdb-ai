import os
from urllib.parse import urlparse

import psycopg2


def connect_mdb():
    mdb_str = os.getenv("MDB")
    url = urlparse(mdb_str)
    return psycopg2.connect(dbname=url.path[1:], user=url.username,
                            password=url.password, host=url.hostname)
