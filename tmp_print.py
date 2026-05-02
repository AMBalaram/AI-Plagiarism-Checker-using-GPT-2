import pathlib
lines = pathlib.Path("app.py").read_text(encoding="utf-8", errors="replace").splitlines()
for i in range(193,206):
    print(f"{i+1:4d}: {lines[i]!r}")
