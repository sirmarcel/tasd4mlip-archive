"""Nested lists of strings -> a `booktabs` tabular.

Formatting a number is the caller's job (`rounder`), so the digits a table shows
are visible in the script that made it. Both documents load `booktabs`.

    from sadmof.tbx import savefile
    from sadmof.tbx.tables import map_across, rounder, tabular

    body = map_across(rows, [str, rounder(2), rounder(1, "e")])
    savefile(tabular(["system", "error", "time"], body), "figures/cost.tex")
"""

from copy import deepcopy
from functools import partial

__all__ = [
    "rounder",
    "to_num",
    "num_rounder",
    "aligner",
    "map_across",
    "table_to_string",
    "table_to_tex",
    "tabular",
]


# ----
# Public API
# ----


def rounder(digits, typ="f"):
    """Formatter for a fixed number of digits: `rounder(2)(1.234) == '1.23'`."""

    def formatter(number):
        if number is None:
            return "--"
        return f"{number:.{digits}{typ}}"

    return formatter


def to_num(entry):
    """Wrap an already-formatted number in `siunitx`'s `\\num{}`."""
    entry = str(entry)
    return entry if entry == "--" else r"\num{" + entry + "}"


def num_rounder(digits, typ="f"):
    """`rounder` with the result wrapped in `\\num{}`; `--` stays a plain dash."""
    formatter = rounder(digits, typ)

    def num_formatter(number):
        return to_num(formatter(number))

    return num_formatter


def aligner(width):
    def formatter(s):
        return f"{s:>{width}}"

    return formatter


def map_across(table, fns):
    """Apply a formatter per column (or one formatter to every cell)."""
    if not callable(fns) and not isinstance(fns, list):
        raise TypeError(f"fns must be a callable or a list, got {type(fns)}")
    if callable(fns):
        return [[fns(entry) for entry in row] for row in table]
    return [[fns[i](entry) for i, entry in enumerate(row)] for row in table]


def table_to_string(data, rownames=None, colnames=None, width=10, colsep="|", rowsep="\n"):
    """Render a table as aligned text; `table_to_tex` is the `&`/`\\\\` variant.

    A row given as a bare string (e.g. `"\\addlinespace"`) is emitted verbatim
    on its own line, without cells or the row separator.
    """
    data = deepcopy(data)
    if rownames is not None:
        for row, name in zip(data, rownames, strict=True):
            row.insert(0, name)
    if colnames is not None:
        data.insert(0, list(colnames))

    ncols = len(next(row for row in data if not isinstance(row, str)))
    widths = [width] * ncols if isinstance(width, int) else list(width)
    formatters = [aligner(w) for w in widths]

    out = ""
    for row in data:
        if isinstance(row, str):
            out += row + "\n"
        else:
            cells = [fmt(entry) for fmt, entry in zip(formatters, row, strict=True)]
            out += f" {colsep} ".join(cells) + rowsep
    return out


table_to_tex = partial(table_to_string, colsep=" & ", rowsep=" \\\\ \n")


def tabular(titles, body, layout=None, width=16, heading=None, blocks=None):
    """A full `booktabs` tabular.

    Args:
        titles: Header cells.
        body: Rows of cells; ignored when `blocks` is given.
        layout: Column spec, e.g. `"l|rrr"`. Defaults to all-right-aligned.
        width: Cell width for the source alignment, int or per-column list.
        heading: Raw TeX inserted above the header row (e.g. `\\cmidrule`
            groups).
        blocks: Groups of rows, separated by `\\midrule`.

    Returns:
        The tabular environment as a string.
    """
    if layout is None:
        layout = " ".join(["r"] * len(titles))
    if blocks is None:
        blocks = [body]

    out = r"\begin{tabular}{" + layout + "}\n" + "\\toprule\n"
    if heading is not None:
        out += heading
    out += table_to_tex([list(titles)], width=width)
    for block in blocks:
        out += "\\midrule\n" + table_to_tex(block, width=width)
    return out + "\\bottomrule\n" + r"\end{tabular}" + "\n"
