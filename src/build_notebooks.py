"""Convert the percent-format sources in src/nb/*.py into Jupyter notebooks and (optionally) execute them.

Usage:  python src/build_notebooks.py [--execute] [name-filter]
"""
import sys
import time
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "nb"
NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)


def convert(py):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if not text:
            return
        if kind == "md":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.split("\n"))
            cells.append(new_markdown_cell(text))
        else:
            cells.append(new_code_cell(text))

    for line in py.read_text(encoding="utf-8").split("\n"):
        if line.startswith("# %%"):
            flush()
            kind, buf = ("md" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


if __name__ == "__main__":
    execute = "--execute" in sys.argv
    flt = [a for a in sys.argv[1:] if not a.startswith("--")]
    for py in sorted(SRC.glob("*.py")):
        if flt and not any(f in py.name for f in flt):
            continue
        nb = convert(py)
        out = NB / (py.stem + ".ipynb")
        if execute:
            from nbconvert.preprocessors import ExecutePreprocessor
            t0 = time.time()
            ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB)}})
            print(f"executed {out.name} in {time.time() - t0:.0f}s")
        nbformat.write(nb, out)
        print("wrote", out)
