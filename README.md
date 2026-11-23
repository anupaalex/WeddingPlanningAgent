python arctic_ingest.py smoke                      # confirm fields and paging work
python arctic_ingest.py volume                     # row counts per year, so you can size the crawl
python arctic_ingest.py crawl --kind posts
python arctic_ingest.py crawl --kind comments
python arctic_ingest.py build                      # raw cache → clean → DuckDB → stats report






pip install -e ".[dev]"
python -m compileall src tools tests -q
pytest
python -m wedding_rag.evaluation.run_log "post-restructure" --profile bge_small_comment_notitle --mode hybrid
python -m wedding_rag.evaluation.run_log --show