import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.db")