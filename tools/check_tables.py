# check_tables.py
import duckdb
from wedding_rag.rag import rag_baseline as rb

print("rag_baseline resolves CHUNKS_TABLE:", rb.CHUNKS_TABLE)
print("rag_baseline resolves INDEX_DIR   :", rb.INDEX_DIR)

con = duckdb.connect(rb.DB_PATH, read_only=True)
tables = con.execute("SELECT table_name FROM information_schema.tables "
                     "WHERE table_name LIKE 'chunks%' ORDER BY 1").fetchall()
for (t,) in tables:
    n, share = con.execute(f"SELECT count(*), avg(CASE WHEN text LIKE 'Thread: %' THEN 1 ELSE 0 END) "
                           f"FROM {t}").fetchone()
    print(f"{t:<58} rows={n:<6} share_with_title={share}")