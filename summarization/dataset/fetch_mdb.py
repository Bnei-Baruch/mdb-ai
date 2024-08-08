from summarization.helper import connect_mdb

QUERY = """
SELECT f_art.uid, f_sum.uid 
    FROM files f_sum
    INNER JOIN content_units cu ON cu.id = f_sum.content_unit_id
    INNER JOIN files f_art ON cu.id = f_art.content_unit_id
    WHERE cu.type_id = 20
    AND f_sum.properties->>'insert_type' = 'summary'
    AND f_sum.language = 'he'
    AND f_sum.published = true
    AND f_art.type = 'text'
    AND f_art.language = 'he'
    AND f_art.published = true
    AND f_art.uid != f_sum.uid;
"""


def run():
    print("Fetching dataset...")
    conn = connect_mdb()
    print("Connected to database")
    cursor = conn.cursor()
    cursor.execute(QUERY)
    res = row_to_dto(cursor.fetchall())
    conn.close()
    print("Fetched dataset")
    return res


def row_to_dto(rows):
    res = []
    for data in rows:
        res.append([data[0], data[1]])
    return res
