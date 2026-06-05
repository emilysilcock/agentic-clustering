"""Generate the LaTeX results table from per-method per-dataset meta.json files.

Walks `results/predictions/<method>/<dataset>/seed=*.meta.json` and aggregates
metrics (mean across seeds). Missing (method, dataset) cells render as ``--''.

The table has two panels:

* **Given $k$** --- methods that take the gold class count as input.
* **Discover $k$** --- methods that decide $k$ themselves. BERTopic and our
  method appear in both (with separate runs); TopicGPT only here, since it
  has no native given-$k$ mode (SPEC \xa75.5).

Best-cell underlining is computed *within each panel* so the highlight is
comparing like with like.

Run after new predictions land:

    uv run --native-tls python -m benchmarking.data_processing.build_results_table

Writes to `paper/results_table.tex`.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from functools import lru_cache

from benchmarking.paths import DATA_DERIVED, RESULTS, ROOT


@dataclass(frozen=True)
class MethodDisplay:
    key: str        # subdirectory name under results/predictions/
    display: str    # row label in the table


# Order within each list = row order within the panel. Methods without
# prediction directories still render (as -- cells) so the table reflects
# the planned baseline set rather than only what has been run so far.
METHODS_GIVEN_K: list[MethodDisplay] = [
    MethodDisplay("lda",                       "LDA"),
    MethodDisplay("sbert_kmeans",              "SBERT+$k$-means"),
    MethodDisplay("bertopic",                  "BERTopic"),
    MethodDisplay("openai_embedding_kmeans",   "LLM-embedding+$k$-means"),
    MethodDisplay("clusterllm",                "ClusterLLM"),
    MethodDisplay("agentic_clustering",        "Agentic clustering (ours)"),
]

METHODS_DISCOVER_K: list[MethodDisplay] = [
    MethodDisplay("bertopic_discoverk",        "BERTopic"),
    MethodDisplay("topicgpt",                  "TopicGPT"),
    MethodDisplay("huang_he",                  "Huang \\& He"),
    MethodDisplay("agentic_clustering_discoverk", "Agentic clustering (ours)"),
]


@dataclass(frozen=True)
class DatasetDisplay:
    key: str        # subdirectory name (matches data/derived/<key>)
    display: str    # column header (full name)


# Order matches Table~\ref{tab:datasets}.
DATASETS: list[DatasetDisplay] = [
    DatasetDisplay("banking77",          "Banking77"),
    DatasetDisplay("clinc150",           "CLINC150"),
    DatasetDisplay("massive_intent",     "MASSIVE-Intent"),
    DatasetDisplay("massive_domain",     "MASSIVE-Domain"),
    DatasetDisplay("goemotions",         "GoEmotions"),
    DatasetDisplay("twenty_newsgroups",  "20 Newsgroups"),
    DatasetDisplay("stackexchange",      "StackExchange"),
]


METRICS = ("ari", "nmi", "acc")


def load_metrics(method_key: str, dataset_key: str) -> dict | None:
    """Metrics for this (method, dataset), averaged across seeds if more than one.

    Returns None when no meta.json exists, so the caller can render a missing cell.
    """
    cell_dir = RESULTS / "predictions" / method_key / dataset_key
    if not cell_dir.exists():
        return None
    meta_files = sorted(cell_dir.glob("seed=*.meta.json"))
    if not meta_files:
        return None

    collected: dict[str, list[float]] = {name: [] for name in METRICS}
    for path in meta_files:
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        m = data.get("metrics", {})
        for name in METRICS:
            v = m.get(name)
            if v is not None:
                collected[name].append(float(v))

    return {name: (statistics.mean(vs) if vs else None) for name, vs in collected.items()}


@lru_cache(maxsize=None)
def gold_k(dataset_key: str) -> int | None:
    """Gold (in-scope) k for a dataset, read from data/derived/<ds>/meta.json."""
    path = DATA_DERIVED / dataset_key / "meta.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fp:
        return json.load(fp).get("k_in_scope")


def predicted_k(method_key: str, dataset_key: str) -> int | None:
    """Reported $\\hat{k}$ for one (method, dataset) cell: populated clusters only.

    Per SPEC §5.5 (decided 2026-05-25), $\\hat{k}$ is the number of clusters with
    **at least one document assigned**, counted from the predictions JSONL —
    *not* the size of the generated taxonomy. Empty taxonomy entries that no
    document is classified into are not counted. This is applied consistently
    across all methods, so we ignore the per-method ``k_actual`` /
    ``n_topics_actual`` fields in meta.json (which record taxonomy size,
    including empty entries) and always count distinct in-scope cluster IDs
    (≥ 0) in the JSONL. -1 (none/unassigned) is excluded so the count matches
    the "in-scope" framing used for gold k.
    """
    cell_dir = RESULTS / "predictions" / method_key / dataset_key
    if not cell_dir.exists():
        return None
    seed0_jsonl = cell_dir / "seed=0.jsonl"
    if not seed0_jsonl.exists():
        return None
    ids: set[int] = set()
    with open(seed0_jsonl, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("predicted_cluster_id")
            if cid is not None and int(cid) >= 0:
                ids.add(int(cid))
    return len(ids) if ids else None


def _fmt(v: float | None, is_best: bool) -> str:
    if v is None:
        return "--"
    s = f"{v:.2f}"
    return f"\\underline{{{s}}}" if is_best else s


def _fmt_int(v: int | None) -> str:
    return "--" if v is None else str(int(v))


def _fmt_cost(v: tuple[float, float] | None) -> str:
    """Format one method's total $-cost for a tight column.

    ``v`` is ``(subscription_usd, total_usd)`` summed across datasets. Methods
    that used the Claude Code Max subscription (subscription > 0) render the
    two parts split out --- e.g. ``$100 + $22`` (flat subscription + metered
    API) --- so the fixed and variable costs are legible. Everything else
    renders as a single total.
    """
    if v is None:
        return "--"
    subscription, total = v
    if subscription > 0:
        api = total - subscription
        return f"\\${subscription:.0f} + \\${api:.0f}"
    if total == 0:
        return "\\$0"
    if total < 0.01:
        return "<\\$0.01"
    if total < 100:
        return f"\\${total:.2f}"
    return f"\\${total:.0f}"


def total_cost(
    method_key: str, datasets: list["DatasetDisplay"]
) -> tuple[float, float] | None:
    """Sum ``(subscription_usd, usd)`` across the given datasets for one method.

    Returns ``(subscription_total, grand_total)``, or None when *no* cell for
    this method exists on disk (so the cost column renders as ``--`` for
    entirely-unrun baselines). ``grand_total`` sums ``cost.usd`` (subscription
    + metered API); ``subscription_total`` sums ``cost.subscription_usd`` (the
    Claude Code Max portion, 0 for methods that don't use it). The renderer
    splits the two as ``$100 + $22`` for subscription methods. Methods with
    partial coverage are summed over whatever cells they have.
    """
    subscription = 0.0
    total = 0.0
    saw_any = False
    for d in datasets:
        cell_dir = RESULTS / "predictions" / method_key / d.key
        if not cell_dir.exists():
            continue
        meta_path = cell_dir / "seed=0.meta.json"
        if not meta_path.exists():
            continue
        saw_any = True
        c = json.loads(meta_path.read_text(encoding="utf-8")).get("cost", {})
        v = c.get("usd")
        if v is not None:
            total += float(v)
        subscription += float(c.get("subscription_usd", 0.0) or 0.0)
    return (subscription, total) if saw_any else None


def _panel_rows(
    methods: list[MethodDisplay],
    datasets: list["DatasetDisplay"],
    cell_metrics: dict[tuple[str, str], dict | None],
    *,
    cost_column: bool,
    cost_lookup: dict[str, tuple[float, float] | None] | None = None,
) -> list[str]:
    """Render one panel for one deck. ``cost_column=True`` appends a Cost
    column at the end of each row using ``cost_lookup``."""
    best: dict[tuple[str, str], float] = {}
    for d in datasets:
        for name in METRICS:
            values = [
                cell_metrics[(m.key, d.key)][name]
                for m in methods
                if cell_metrics[(m.key, d.key)] is not None
                and cell_metrics[(m.key, d.key)][name] is not None
            ]
            if values:
                best[(d.key, name)] = max(values)

    rows: list[str] = []
    for m in methods:
        cells = [m.display]
        for d in datasets:
            cells.append(_fmt_int(predicted_k(m.key, d.key)))
            metrics = cell_metrics[(m.key, d.key)]
            if metrics is None:
                cells.extend(["--", "--", "--"])
            else:
                for name in METRICS:
                    v = metrics[name]
                    is_best = v is not None and v == best.get((d.key, name))
                    cells.append(_fmt(v, is_best))
        if cost_column:
            cells.append(_fmt_cost((cost_lookup or {}).get(m.key)))
        rows.append("    " + " & ".join(cells) + " \\\\")
    return rows


def _deck_lines(
    datasets: list[DatasetDisplay],
    *,
    cell_metrics: dict[tuple[str, str], dict | None],
    cost_column: bool,
    cost_lookup: dict[str, float | None] | None,
) -> list[str]:
    """Emit one tabular deck covering the given subset of datasets.

    ``cost_column`` adds a final $-total column (used on the bottom deck only
    so the figure carries cost once, not twice).
    """
    n_data = len(datasets)
    cells_per_dataset = 4  # k_pred, ARI, NMI, ACC

    # Column spec: method label + (4 cols per dataset) + optional cost.
    col_spec = "l" + (" cccc" * n_data) + (" r" if cost_column else "")

    # Top header row: dataset names as 4-wide multicolumns, plus optional
    # 1-wide Cost header.
    top_header_cells = [""] + [
        f"\\multicolumn{{{cells_per_dataset}}}{{c}}{{\\textbf{{{d.display}}}}}"
        for d in datasets
    ]
    if cost_column:
        top_header_cells.append("\\textbf{Cost}")
    top_header = " & ".join(top_header_cells) + " \\\\"

    # cmidrule under each dataset multicolumn header.
    cmid_parts = []
    for i in range(n_data):
        first = 2 + i * cells_per_dataset
        last = first + cells_per_dataset - 1
        cmid_parts.append(f"\\cmidrule(lr){{{first}-{last}}}")
    if cost_column:
        col = 2 + n_data * cells_per_dataset
        cmid_parts.append(f"\\cmidrule(lr){{{col}-{col}}}")
    cmid_line = "".join(cmid_parts)

    # Sub-header row: Method + ($\hat{k}$ ARI NMI ACC) × n_data [+ total $].
    sub_header_cells = ["\\textbf{Method}"] + ["$\\hat{k}$", "ARI", "NMI", "ACC"] * n_data
    if cost_column:
        sub_header_cells.append("total \\$")
    sub_header = " & ".join(sub_header_cells) + " \\\\"

    # Panel header spans the full deck width.
    panel_span = 1 + cells_per_dataset * n_data + (1 if cost_column else 0)

    def panel_header(title: str) -> str:
        return (
            f"    \\multicolumn{{{panel_span}}}{{l}}"
            f"{{\\textit{{{title}}}}} \\\\"
        )

    deck = [
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    {top_header}",
        f"    {cmid_line}",
        f"    {sub_header}",
        "    \\midrule",
        panel_header("Given $k$"),
        "    \\midrule",
    ]
    deck.extend(_panel_rows(
        METHODS_GIVEN_K, datasets, cell_metrics,
        cost_column=cost_column, cost_lookup=cost_lookup,
    ))
    deck.extend([
        "    \\midrule",
        panel_header("Discover $k$"),
        "    \\midrule",
    ])
    deck.extend(_panel_rows(
        METHODS_DISCOVER_K, datasets, cell_metrics,
        cost_column=cost_column, cost_lookup=cost_lookup,
    ))
    deck.extend([
        "    \\bottomrule",
        "  \\end{tabular}%",
    ])
    return deck


def build_table() -> str:
    # Two-deck split: top deck = 4 datasets, bottom deck = remaining 3 plus a
    # total-cost column. Datasets are sliced in their DATASETS-list order so
    # the deck split is deterministic with the dataset table in §4.
    top_datasets = DATASETS[:4]
    bottom_datasets = DATASETS[4:]

    all_methods = METHODS_GIVEN_K + METHODS_DISCOVER_K
    cell_metrics: dict[tuple[str, str], dict | None] = {
        (m.key, d.key): load_metrics(m.key, d.key)
        for m in all_methods for d in DATASETS
    }
    cost_lookup: dict[str, tuple[float, float] | None] = {
        m.key: total_cost(m.key, DATASETS) for m in all_methods
    }

    # Both decks must be scaled by the *same* factor, otherwise resizing each
    # independently to \textwidth gives them different font sizes (the top deck
    # has more dataset columns and so shrinks more). We measure both decks into
    # boxes, scale the wider (top) deck to \textwidth, and scale the bottom deck
    # by that same factor via \fpeval{wd_bottom/wd_top}. The bottom deck then
    # ends up slightly narrower than the full width and is centred. The
    # \newsavebox declarations are guarded so a stray double \input is safe.
    lines = [
        "% Auto-generated by benchmarking/data_processing/build_results_table.py.",
        "% Do not edit by hand --- re-run the builder after new predictions land.",
        "\\ifdefined\\acdeckone\\else\\newsavebox{\\acdeckone}\\fi",
        "\\ifdefined\\acdecktwo\\else\\newsavebox{\\acdecktwo}\\fi",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\sbox{\\acdeckone}{%",
    ]
    lines.extend(_deck_lines(
        top_datasets,
        cell_metrics=cell_metrics,
        cost_column=False,
        cost_lookup=None,
    ))
    lines.append("  }")
    lines.append("  \\sbox{\\acdecktwo}{%")
    lines.extend(_deck_lines(
        bottom_datasets,
        cell_metrics=cell_metrics,
        cost_column=True,
        cost_lookup=cost_lookup,
    ))
    lines.append("  }")
    # Top deck sets the scale; bottom deck is scaled by the same factor so the
    # two share one font size. A vertical gap separates the stacked tabulars.
    lines.append("  \\resizebox{\\textwidth}{!}{\\usebox{\\acdeckone}}")
    lines.append("")
    lines.append("  \\vspace{0.5em}")
    lines.append("")
    lines.append(
        "  \\resizebox{\\fpeval{\\wd\\acdecktwo/\\wd\\acdeckone}\\textwidth}{!}"
        "{\\usebox{\\acdecktwo}}"
    )
    lines.extend([
        "  \\caption{Clustering results across seven benchmarks. Cost is the "
        "total USD across all seven datasets; methods that use the Claude Code "
        "Max subscription show it as the flat \\$100 subscription plus metered "
        "API spend (\\$100 + API).}",
        "  \\label{tab:results}",
        "\\end{table*}",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    out_path = ROOT / "paper" / "results_table.tex"
    out_path.write_text(build_table(), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
