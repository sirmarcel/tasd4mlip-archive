"""Names, colors, and line styles shared by every figure and table.

One place to change how a model or a structure looks, so panels agree across
figures. Strings are LaTeX (the style file turns `text.usetex` on) and must
compile under the preamble in `plots.mplstyle`.
"""

__all__ = [
    "textwidth",
    "nolabel",
    "tol_vibrant",
    "tol_muted",
    "black",
    "red",
    "teal",
    "orange",
    "blue",
    "cyan",
    "magenta",
    "grey",
    "darkgrey",
    "solid",
    "dashed",
    "dotted",
    "dashdot",
    "loosedot",
    "finedot",
    "finedash",
    "cross",
    "diamond",
    "star",
    "dot",
    "bigdot",
    "square",
    "plus",
    "model_names",
    "model_names_short",
    "model_colors",
    "model_linestyles",
    "model_markers",
    "model_order",
    "structure_names",
    "structure_colors",
]

# NeurIPS 2026 (`\usepackage[preprint]{neurips_2026}`): `textwidth=5.5in`.
textwidth = 5.5

nolabel = "__nolabel__"

# Paul Tol's qualitative schemes: colorblind-safe, print-safe.
tol_vibrant = [
    "#EE7733",
    "#0077BB",
    "#33BBEE",
    "#EE3377",
    "#CC3311",
    "#009988",
    "#BBBBBB",
    "#000000",
]

tol_muted = [
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#332288",
    "#DDCC77",
    "#999933",
    "#CC6677",
    "#882255",
    "#AA4499",
    "#DDDDDD",
]

black = "#000000"
red = "#CC3311"
teal = "#009988"
orange = "#EE7733"
blue = "#0077BB"
cyan = "#33BBEE"
magenta = "#EE3377"
grey = "#BBBBBB"
darkgrey = "#888888"

solid = "solid"
dashed = "dashed"
dotted = "dotted"
dashdot = "dashdot"
loosedot = (0, (1, 3))
finedot = (0, (0.5, 2))
finedash = (0, (4, 3))

cross = "x"
diamond = "D"
star = "*"
dot = "."
bigdot = "o"
square = "s"
plus = "P"


# ----
# Models
# ----

# Keys are the `--model` values of the work/ experiments.
model_order = ["mace", "pet-xs", "pet-s"]

model_names = {
    "mace": r"\textsc{Mace}-MP-0 {\small (M)}",
    "pet-xs": r"\textsc{Pet}-MAD-XS",
    "pet-s": r"\textsc{Pet}-MAD-S",
}

model_names_short = {
    "mace": r"\textsc{Mace}",
    "pet-xs": r"\textsc{Pet}-XS",
    "pet-s": r"\textsc{Pet}-S",
}

model_colors = {
    "mace": blue,
    "pet-xs": orange,
    "pet-s": teal,
}

model_linestyles = {
    "mace": solid,
    "pet-xs": dashed,
    "pet-s": dotted,
}

# Keeps figures legible in grayscale: hue alone must never carry the model.
model_markers = {
    "mace": bigdot,
    "pet-xs": square,
    "pet-s": diamond,
}


# ----
# Structures
# ----

structure_names = {
    "mof210": "MOF-210",
    "mil100": "MIL-100",
    "mil101": "MIL-101",
    "mof177": "MOF-177",
}

structure_colors = {
    "mof210": tol_muted[3],
    "mil100": tol_muted[1],
    "mil101": tol_muted[6],
    "mof177": tol_muted[4],
}
