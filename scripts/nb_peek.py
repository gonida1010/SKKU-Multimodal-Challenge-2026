"""Peek into a Jupyter notebook without loading the whole (possibly huge) file into context.

Usage:
  python scripts/nb_peek.py <notebook.ipynb>            # cell map: index, type, line count, first line
  python scripts/nb_peek.py <notebook.ipynb> 8          # print full source of cell 8
  python scripts/nb_peek.py <notebook.ipynb> 8,9,11     # print full source of several cells
  python scripts/nb_peek.py <notebook.ipynb> grep TEXT  # print cells whose source contains TEXT
"""
import json, sys

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def main():
    path = sys.argv[1]
    nb = load(path)
    cells = nb["cells"]

    if len(sys.argv) >= 4 and sys.argv[2] == "grep":
        needle = sys.argv[3]
        for i, c in enumerate(cells):
            src = "".join(c.get("source", []))
            if needle in src:
                print(f"===== CELL {i} ({c['cell_type']}) contains {needle!r} =====")
                print(src)
                print()
        return

    if len(sys.argv) >= 3:
        for part in sys.argv[2].split(","):
            i = int(part)
            c = cells[i]
            print(f"===== CELL {i} ({c['cell_type']}) =====")
            print("".join(c.get("source", [])))
            print()
        return

    print(f"{len(cells)} cells in {path}")
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        lines = src.splitlines()
        first = next((ln for ln in lines if ln.strip()), "")
        print(f"[{i:2d}] {c['cell_type']:8s} {len(lines):4d}L | {first[:90]}")

if __name__ == "__main__":
    main()
