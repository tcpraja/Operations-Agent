import base64
import io
import logging
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from openpyxl import load_workbook

from app.rag import (
    KNOWLEDGE_DIR,
    load_document,
)


logger = logging.getLogger(__name__)


# =========================================================
# HELPERS
# =========================================================

def _figure_to_base64_png(figure) -> str:

    buffer = io.BytesIO()

    figure.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        dpi=120,
    )

    plt.close(figure)

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return f"data:image/png;base64,{encoded}"


SEVERITY_COLORS = {
    "LOW": "#2e7d32",
    "MEDIUM": "#f9a825",
    "HIGH": "#ef6c00",
    "CRITICAL": "#c62828",
}

SEVERITY_ORDER = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
]


# =========================================================
# SEVERITY / CONFIDENCE DASHBOARD
# =========================================================

def build_severity_dashboard(
    severity: str,
    confidence: float,
    escalation_required: bool,
) -> str | None:
    """
    Render a small dashboard summarizing severity,
    confidence and escalation status as a PNG image
    (base64 data URI).
    """

    try:

        figure, (
            severity_axis,
            confidence_axis,
        ) = plt.subplots(
            1,
            2,
            figsize=(7, 2.6),
        )

        # -----------------------------------------------------
        # SEVERITY BAR
        # -----------------------------------------------------

        levels = SEVERITY_ORDER

        heights = [
            1 if level == severity else 0.25
            for level in levels
        ]

        colors = [
            SEVERITY_COLORS.get(
                level,
                "#9e9e9e",
            )
            for level in levels
        ]

        severity_axis.bar(
            levels,
            heights,
            color=colors,
        )

        severity_axis.set_ylim(0, 1.2)
        severity_axis.set_yticks([])
        severity_axis.set_title(
            f"Severity: {severity}"
            + (
                "  (ESCALATE)"
                if escalation_required
                else ""
            ),
            fontsize=10,
        )

        # -----------------------------------------------------
        # CONFIDENCE GAUGE (horizontal bar)
        # -----------------------------------------------------

        clamped_confidence = max(
            0.0,
            min(1.0, confidence or 0.0),
        )

        confidence_axis.barh(
            [""],
            [clamped_confidence],
            color="#1565c0",
        )

        confidence_axis.barh(
            [""],
            [1 - clamped_confidence],
            left=[clamped_confidence],
            color="#e0e0e0",
        )

        confidence_axis.set_xlim(0, 1)
        confidence_axis.set_yticks([])
        confidence_axis.set_title(
            f"Confidence: "
            f"{clamped_confidence * 100:.0f}%",
            fontsize=10,
        )

        figure.tight_layout()

        return _figure_to_base64_png(
            figure
        )

    except Exception:

        logger.exception(
            "severity_dashboard_render_failed"
        )

        return None


# =========================================================
# HISTORICAL FAILURE TREND CHART
# =========================================================

FAILURE_KEYWORDS = [
    "motor failure",
    "bearing",
    "blockage",
    "blocked",
    "misalignment",
    "vibration",
    "blade",
]

YEAR_PATTERN = re.compile(
    r"\b(20\d{2})\b"
)


def build_failure_trend_chart(
    retrieved_sources: list[str],
) -> str | None:
    """
    Build a grouped bar chart comparing failure-keyword
    frequency across knowledge documents that have a
    year in their filename (e.g. yearly benchmarking
    reports). Returns None when fewer than two distinct
    years are available, since a trend needs more than
    one data point.
    """

    sources_by_year: dict[str, list[str]] = {}

    for source in retrieved_sources:

        match = YEAR_PATTERN.search(source)

        if not match:
            continue

        year = match.group(1)

        sources_by_year.setdefault(
            year,
            [],
        ).append(source)

    if len(sources_by_year) < 2:
        return None

    try:

        year_keyword_counts: dict[
            str,
            dict[str, int],
        ] = {}

        for year, sources in sources_by_year.items():

            combined_text = ""

            for source in sources:

                file_path = (
                    KNOWLEDGE_DIR / source
                )

                if not file_path.exists():
                    continue

                combined_text += (
                    load_document(
                        file_path
                    ).lower()
                )

            year_keyword_counts[year] = {
                keyword: combined_text.count(
                    keyword
                )
                for keyword in FAILURE_KEYWORDS
            }

        years = sorted(
            year_keyword_counts.keys()
        )

        figure, axis = plt.subplots(
            figsize=(7, 3.2)
        )

        bar_width = 0.8 / len(years)

        x_positions = range(
            len(FAILURE_KEYWORDS)
        )

        for index, year in enumerate(years):

            offsets = [
                x
                + index * bar_width
                - (
                    bar_width
                    * (len(years) - 1)
                    / 2
                )
                for x in x_positions
            ]

            counts = [
                year_keyword_counts[year][
                    keyword
                ]
                for keyword in FAILURE_KEYWORDS
            ]

            axis.bar(
                offsets,
                counts,
                width=bar_width,
                label=year,
            )

        axis.set_xticks(
            list(x_positions)
        )

        axis.set_xticklabels(
            FAILURE_KEYWORDS,
            rotation=30,
            ha="right",
            fontsize=8,
        )

        axis.set_ylabel(
            "Mentions"
        )

        axis.set_title(
            "Failure-mode mentions by year "
            "(matched knowledge documents)",
            fontsize=10,
        )

        axis.legend(
            fontsize=8
        )

        figure.tight_layout()

        return _figure_to_base64_png(
            figure
        )

    except Exception:

        logger.exception(
            "failure_trend_chart_render_failed"
        )

        return None


# =========================================================
# FAILURE COUNT BY TYPE (from structured benchmarking data)
# =========================================================

FAILURE_SECTION_HEADER = "failures observed"

MAX_FAILURE_TYPES_SHOWN = 15


def _extract_failure_counts_from_sheet(
    rows: list[tuple],
) -> dict[str, int]:
    """
    Given the raw rows of a benchmarking sheet, look for a
    "Failures observed" section header followed by a row of
    failure-type names, then count how many site rows have
    a mark in each failure-type column.

    Returns an empty dict if the sheet does not contain
    this structure.
    """

    if len(rows) < 3:
        return {}

    section_header_row = rows[0]

    section_start = None

    for index, cell in enumerate(
        section_header_row
    ):

        if (
            isinstance(cell, str)
            and cell.strip().lower()
            == FAILURE_SECTION_HEADER
        ):

            section_start = index
            break

    if section_start is None:
        return {}

    column_names_row = rows[1]

    failure_columns = [
        (index, str(name).strip())
        for index, name in enumerate(
            column_names_row
        )
        if index >= section_start
        and name
        and str(name).strip()
    ]

    if not failure_columns:
        return {}

    counts: dict[str, int] = {
        name: 0
        for _, name in failure_columns
    }

    for row in rows[2:]:

        if (
            row
            and isinstance(row[0], str)
            and "total" in row[0].strip().lower()
        ):
            continue

        for index, name in failure_columns:

            if (
                index < len(row)
                and row[index] not in (None, "")
            ):
                counts[name] += 1

    return {
        name: count
        for name, count in counts.items()
        if count > 0
    }


def get_failure_counts(
    retrieved_sources: list[str],
) -> dict[str, int]:
    """
    Scan the retrieved xlsx knowledge sources for a
    structured "Failures observed" matrix (site rows x
    failure-type columns) and return how many sites report
    each failure type. Returns an empty dict when no
    matching knowledge source has this structure.
    """

    for source in retrieved_sources:

        if not source.lower().endswith(
            (".xlsx", ".xlsm")
        ):
            continue

        file_path = KNOWLEDGE_DIR / source

        if not file_path.exists():
            continue

        try:

            workbook = load_workbook(
                filename=file_path,
                read_only=True,
                data_only=True,
            )

            counts: dict[str, int] = {}

            try:

                for worksheet in (
                    workbook.worksheets
                ):

                    rows = list(
                        worksheet.iter_rows(
                            values_only=True
                        )
                    )

                    counts = (
                        _extract_failure_counts_from_sheet(
                            rows
                        )
                    )

                    if counts:
                        break

            finally:

                workbook.close()

            if counts:
                return counts

        except Exception:

            logger.exception(
                "failure_count_scan_failed "
                f"source={source}"
            )

            continue

    return {}


def build_failure_count_chart(
    retrieved_sources: list[str],
) -> str | None:
    """
    Chart how many sites report each failure type, based
    on get_failure_counts(). Returns None when no matching
    knowledge source has this structure.
    """

    counts = get_failure_counts(
        retrieved_sources
    )

    if not counts:
        return None

    try:

        sorted_counts = sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:MAX_FAILURE_TYPES_SHOWN]

        labels = [
            name
            for name, _ in sorted_counts
        ][::-1]

        values = [
            value
            for _, value in sorted_counts
        ][::-1]

        figure, axis = plt.subplots(
            figsize=(
                7,
                0.35 * len(labels) + 1,
            )
        )

        axis.barh(
            labels,
            values,
            color="#6a1b9a",
        )

        axis.set_xlabel(
            "Sites reporting this failure"
        )

        axis.set_title(
            "Failure types observed "
            "across benchmarked sites",
            fontsize=10,
        )

        axis.tick_params(
            axis="y",
            labelsize=8,
        )

        figure.tight_layout()

        return _figure_to_base64_png(
            figure
        )

    except Exception:

        logger.exception(
            "failure_count_chart_render_failed"
        )

        return None


# =========================================================
# 80/20 PARETO ANALYSIS OF FAILURE CAUSES
# =========================================================

PARETO_VITAL_FEW_THRESHOLD = 80.0


def build_pareto_analysis(
    retrieved_sources: list[str],
) -> dict | None:
    """
    Apply the 80/20 (Pareto) principle to the real failure
    counts extracted from the benchmarking knowledge base:
    rank failure types by how many sites reported them,
    and identify the smallest set of failure types
    ("the vital few") that together account for roughly
    80% of all reported failures.

    Returns None when no knowledge source has the
    structured failure-count data needed for this.
    """

    counts = get_failure_counts(
        retrieved_sources
    )

    if not counts:
        return None

    total = sum(counts.values())

    if total <= 0:
        return None

    sorted_counts = sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    items = []

    cumulative = 0

    vital_few_count = 0

    for rank, (cause, count) in enumerate(
        sorted_counts,
        start=1,
    ):

        cumulative += count

        cumulative_percentage = round(
            (cumulative / total) * 100,
            1,
        )

        is_vital_few = (
            cumulative_percentage
            <= PARETO_VITAL_FEW_THRESHOLD
            or vital_few_count == 0
        )

        if is_vital_few:
            vital_few_count += 1

        items.append(
            {
                "rank": rank,
                "cause": cause,
                "count": count,
                "percentage": round(
                    (count / total) * 100,
                    1,
                ),
                "cumulative_percentage": (
                    cumulative_percentage
                ),
                "is_vital_few": is_vital_few,
            }
        )

    summary = (
        f"{vital_few_count} of "
        f"{len(sorted_counts)} failure causes "
        f"({round(vital_few_count / len(sorted_counts) * 100)}%"
        f" of causes) account for "
        f"{items[vital_few_count - 1]['cumulative_percentage']}%"
        f" of all reported failures."
    )

    return {
        "items": items,
        "vital_few_count": vital_few_count,
        "total_causes": len(sorted_counts),
        "summary": summary,
    }


def build_cutter_knowledge_pareto() -> dict | None:
    """Pareto of cutter-related site reports in the latest benchmark workbook."""
    sources = sorted(
        KNOWLEDGE_DIR.glob("*cutter benchmarking*.xlsx"),
        key=lambda path: path.name,
        reverse=True,
    )
    for source in sources:
        counts = get_failure_counts([source.name])
        cutter_counts = {
            name: count for name, count in counts.items()
            if re.search(r"\b(?:cutter|cutting)s?\b", name, re.I)
        }
        if not cutter_counts:
            continue
        ranked = sorted(cutter_counts.items(), key=lambda item: (-item[1], item[0]))
        total = sum(cutter_counts.values())
        cumulative = 0
        items = []
        for rank, (category, count) in enumerate(ranked, 1):
            cumulative += count
            items.append({
                "rank": rank,
                "category": category,
                "count": count,
                "cumulative_percentage": round(100 * cumulative / total, 1),
            })

        figure, axis = plt.subplots(figsize=(max(8, len(items) * 0.9), 5))
        positions = list(range(len(items)))
        axis.bar(positions, [item["count"] for item in items], color="#0086a8")
        axis.set_ylabel("Benchmarked sites reporting category")
        axis.set_xticks(positions, [item["category"] for item in items], rotation=45, ha="right", fontsize=8)
        axis.set_title("Cutter-related failure reports by site")
        percent_axis = axis.twinx()
        percent_axis.plot(positions, [item["cumulative_percentage"] for item in items], color="#e45d38", marker="o")
        percent_axis.axhline(80, color="#777777", linestyle="--", linewidth=1)
        percent_axis.set_ylim(0, 110)
        percent_axis.set_ylabel("Cumulative share of reports (%)")
        figure.tight_layout()
        image = _figure_to_base64_png(figure)
        plt.close(figure)
        return {
            "source": source.name,
            "sheet": "Cutters",
            "items": items,
            "total_reports": total,
            "image": image,
            "note": "Counts are benchmarked sites reporting each failure category, not trip event counts or confirmed causes of cutter trips.",
        }
    return None


def build_custom_chart(
    chart_type: str,
    title: str,
    labels: list[str],
    values: list[float],
) -> str | None:
    """
    Build a trend (line) chart or pie chart from user-supplied
    or conversation-supplied labels/values. Returns None if the
    data is unusable (empty, mismatched lengths, non-positive
    values for a pie chart).
    """

    if not labels or not values or len(labels) != len(values):
        return None

    try:
        values = [float(value) for value in values]
    except (TypeError, ValueError):
        return None

    try:

        if chart_type == "pie":

            if any(value <= 0 for value in values):
                return None

            figure, axis = plt.subplots(figsize=(6, 6))
            axis.pie(
                values,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
            )
            axis.set_title(title, fontsize=12)
            figure.tight_layout()
            return _figure_to_base64_png(figure)

        # Default: trend / line chart.
        figure, axis = plt.subplots(figsize=(7, 3.5))
        axis.plot(labels, values, marker="o", color="#0086a8")
        axis.set_title(title, fontsize=12)
        axis.set_ylabel("Value")
        axis.tick_params(axis="x", rotation=30, labelsize=9)
        figure.tight_layout()
        return _figure_to_base64_png(figure)

    except Exception:

        logger.exception("custom_chart_render_failed chart_type=%s", chart_type)

        return None


def build_pareto_from_counts(
    counts: dict[str, int], source: str, metric: str, note: str
) -> dict | None:
    """Render a Pareto from supplied category counts without inferring causes."""
    ranked = sorted(
        ((str(name).strip(), int(count)) for name, count in counts.items() if int(count) > 0),
        key=lambda item: (-item[1], item[0].lower()),
    )
    if not ranked:
        return None
    total = sum(count for _, count in ranked)
    cumulative = 0
    items = []
    for rank, (category, count) in enumerate(ranked, 1):
        cumulative += count
        items.append({
            "rank": rank,
            "category": category,
            "count": count,
            "cumulative_percentage": round(cumulative * 100 / total, 1),
        })
    figure, axis = plt.subplots(figsize=(max(8, len(items) * 0.9), 5))
    positions = list(range(len(items)))
    axis.bar(positions, [item["count"] for item in items], color="#0086a8")
    axis.set_ylabel(metric)
    axis.set_xticks(positions, [item["category"] for item in items], rotation=45, ha="right", fontsize=8)
    axis.set_title("Pareto analysis")
    percent_axis = axis.twinx()
    percent_axis.plot(positions, [item["cumulative_percentage"] for item in items], color="#e45d38", marker="o")
    percent_axis.axhline(80, color="#777777", linestyle="--", linewidth=1)
    percent_axis.set_ylim(0, 110)
    percent_axis.set_ylabel("Cumulative share (%)")
    figure.tight_layout()
    image = _figure_to_base64_png(figure)
    return {
        "source": source,
        "sheet": "",
        "metric": metric,
        "items": items,
        "total_reports": total,
        "image": image,
        "note": note,
    }
