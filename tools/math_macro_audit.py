"""Audit every macro used in every math block against GitHub's MathJax whitelist.

GitHub rejects a specific set of macros and renders the whole block as raw source
when it hits one. `\\tag`, `\\label` and `\\eqref` are the well-known ones;
`\\operatorname` is also rejected and is easy to reach for. This lists what is
actually used so the check is exhaustive rather than a guess at which macro broke.
"""

import collections
import pathlib
import re
import sys

ALLOWED = set(
    """frac sum prod int sqrt partial nabla infty pi sigma lambda mu nu alpha beta gamma
    delta epsilon varepsilon theta phi varphi psi omega Sigma Lambda Delta Omega Gamma Phi
    Psi Theta mathbb mathcal mathrm mathbf mathit text textbf le ge ne approx sim equiv
    times cdot cdots ldots dots to rightarrow leftarrow Rightarrow Leftarrow longrightarrow
    Longrightarrow mapsto in notin subset subseteq cup cap setminus emptyset forall exists
    neg land lor implies iff left right big Big bigg Bigg langle rangle lvert rvert lVert
    rVert vert Vert hat bar tilde vec dot max min log exp lim sup inf arg det dim ker deg
    circ ast star quad qquad begin end array matrix pmatrix bmatrix cases aligned align
    overline underline binom geq leq neq pm mp colon mid ge le""".split()
)

BANNED = set(
    """operatorname DeclareMathOperator newcommand def tag label eqref require href url
    includegraphics renewcommand ensuremath color textcolor""".split()
)

fail = False
for name in sys.argv[1:]:
    path = pathlib.Path(name)
    if not path.exists():
        continue
    txt = path.read_text(encoding="utf-8")
    blocks = re.findall(r"\$\$(.*?)\$\$", txt, re.S)
    blocks += re.findall(r"(?<!\$)\$([^$\n]{2,200})\$(?!\$)", txt)
    macros = collections.Counter()
    for b in blocks:
        for m in re.findall(r"\\([A-Za-z]+)", b):
            macros[m] += 1
    bad = {m: c for m, c in macros.items() if m in BANNED}
    unknown = {m: c for m, c in macros.items() if m not in ALLOWED and m not in BANNED}
    print(f"== {name}: {len(blocks)} math blocks, {len(macros)} distinct macros")
    print(f"   BANNED : {bad if bad else 'none'}")
    print(f"   unknown: {unknown if unknown else 'none'}")
    if bad:
        fail = True

sys.exit(1 if fail else 0)
