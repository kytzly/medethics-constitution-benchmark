"""Load the four ethical constitutions, one text file each, from constitutions/.

C0 None | C1 AMA | C2 WHO | C3 Mitigation (safety guardrail).
"""
import os

_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "constitutions")
_FILES = {
    "C0": "C0_none.txt",
    "C1": "C1_ama.txt",
    "C2": "C2_who.txt",
    "C3": "C3_mitigation.txt",
}

CONSTITUTIONS = {
    cid: open(os.path.join(_DIR, fn), encoding="utf-8").read().strip()
    for cid, fn in _FILES.items()
}
