import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Make `plantdisease` importable without installing the package.
sys.path.insert(0, str(ROOT / "src"))
# Make the Gradio app modules importable the same way app.py imports them.
sys.path.insert(0, str(ROOT / "app"))
