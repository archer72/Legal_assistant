import tempfile
import os
from ingestion.ingest import build_index


def test_build_index_runs():
    here = os.path.dirname(__file__)
    data_dir = os.path.join(here, '..', 'sample_data')
    outdir = tempfile.mkdtemp()
    # This test expects small data; it's a smoke test for the ingest flow
    build_index(data_dir, outdir, 'sentence-transformers/all-MiniLM-L6-v2', 400, 50)
    assert os.path.exists(outdir)
