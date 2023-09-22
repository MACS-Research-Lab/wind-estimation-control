import sys
from pathlib import Path

local_path = Path(__file__).absolute().parent.parent.resolve()

if local_path not in sys.path:
    sys.path.append(local_path)
