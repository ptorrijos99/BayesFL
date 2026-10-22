"""Convert textual BIF to XMLBIF, which is what the Java side reads as gold standard.

bnlearn (R) writes BIF 0.15; weka's BIFReader, used by every harness here to load
the gold-standard DAG, wants XMLBIF. pgmpy reads one and writes the other, so the
conversion is a two-line round trip -- the value of this script is that it does it
over a directory and reports what failed, since a handful of bnRep networks carry
CPTs that do not survive the round trip.

Usage (from the repository root):
    python res/networks/generators/bif_to_xbif.py res/networks/BIF res/networks/xbif [--only NAME]
"""
import os
import sys
import glob

from pgmpy.readwrite import BIFReader, XMLBIFWriter


def convert(bif_path, out_dir):
    name = os.path.splitext(os.path.basename(bif_path))[0]
    out_path = os.path.join(out_dir, name + ".xbif")
    model = BIFReader(path=bif_path).get_model()
    XMLBIFWriter(model).write_xmlbif(out_path)
    return name, out_path, len(model.nodes()), len(model.edges())


def main():
    bif_dir, out_dir = sys.argv[1], sys.argv[2]
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]

    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(bif_dir, "*.bif")))
    if only:
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] == only]

    ok, failed = 0, []
    for p in paths:
        try:
            name, out_path, n, e = convert(p, out_dir)
            print(f"OK   {name:<30} {n:4d} nodes {e:4d} arcs -> {out_path}")
            ok += 1
        except Exception as exc:
            print(f"FAIL {os.path.basename(p):<30} {type(exc).__name__}: {exc}")
            failed.append(os.path.basename(p))

    print(f"\nconverted {ok} of {len(paths)}")
    if failed:
        print("failed:", ", ".join(failed))


if __name__ == "__main__":
    main()
