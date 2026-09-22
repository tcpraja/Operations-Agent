"""Small code-native visuals for common equipment requests."""

from xml.sax.saxutils import escape


def build_fishbone_svg(problem: str, categories: dict[str, list[str]]) -> str:
    """Render a Fishbone (Ishikawa) diagram from RCA categories.

    ``categories`` maps a cause category (e.g. Machine, Method,
    Material, Manpower, Measurement, Environment) to a short list
    of contributing causes under it. Only non-empty categories are
    drawn; each is capped to 2 causes to keep the diagram legible.
    """

    active = [
        (name, causes) for name, causes in categories.items() if causes
    ][:6]

    if not active:
        active = [(name, []) for name in list(categories)[:6]]

    branch_count = max(len(active), 1)
    spine_x_start = 70
    spine_length = 150 * branch_count
    width = spine_x_start + spine_length + 260
    height = 460
    spine_y = height // 2

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Fishbone Ishikawa root cause diagram">',
        f'<rect width="{width}" height="{height}" rx="24" fill="#101827"/>',
        f'<text x="42" y="42" fill="#f8fafc" font-family="Arial,sans-serif" '
        f'font-size="24" font-weight="700">Fishbone (Ishikawa) diagram</text>',
        f'<text x="42" y="66" fill="#94a3b8" font-family="Arial,sans-serif" '
        f'font-size="13">Root cause categories for the reported problem</text>',
    ]

    spine_x_end = spine_x_start + spine_length
    parts.append(
        f'<line x1="{spine_x_start}" y1="{spine_y}" x2="{spine_x_end}" '
        f'y2="{spine_y}" stroke="#7dd3fc" stroke-width="5"/>'
    )
    parts.append(
        f'<path d="M{spine_x_end} {spine_y - 12} L{spine_x_end + 34} '
        f'{spine_y} L{spine_x_end} {spine_y + 12} Z" fill="#7dd3fc"/>'
    )

    box_x = spine_x_end + 34
    box_w = width - box_x - 30
    parts.append(
        f'<rect x="{box_x}" y="{spine_y - 50}" width="{box_w}" height="100" '
        f'rx="12" fill="#334155" stroke="#fbbf24" stroke-width="4"/>'
    )
    problem_text = escape(problem[:70])
    parts.append(
        f'<text x="{box_x + box_w / 2}" y="{spine_y + 6}" fill="#fbbf24" '
        f'font-family="Arial,sans-serif" font-size="15" font-weight="700" '
        f'text-anchor="middle">{problem_text}</text>'
    )

    spacing = spine_length / (branch_count + 1)

    for index, (name, causes) in enumerate(active):
        x = spine_x_start + spacing * (index + 1)
        above = index % 2 == 0
        direction = 1 if above else -1
        branch_end_x = x + 55
        branch_end_y = spine_y - direction * 130

        parts.append(
            f'<line x1="{x}" y1="{spine_y}" x2="{branch_end_x}" '
            f'y2="{branch_end_y}" stroke="#38bdf8" stroke-width="4"/>'
        )

        label_y = branch_end_y - direction * 12
        parts.append(
            f'<text x="{branch_end_x}" y="{label_y}" fill="#e2e8f0" '
            f'font-family="Arial,sans-serif" font-size="14" '
            f'font-weight="700" text-anchor="middle">{escape(name)}</text>'
        )

        cause_y = label_y
        for cause in causes[:2]:
            cause_y += direction * 18
            short_cause = cause if len(cause) <= 34 else cause[:31] + "..."
            parts.append(
                f'<text x="{branch_end_x}" y="{cause_y}" fill="#94a3b8" '
                f'font-family="Arial,sans-serif" font-size="11" '
                f'text-anchor="middle">{escape(short_cause)}</text>'
            )

    parts.append("</svg>")
    return "".join(parts)


def build_pump_svg() -> str:
    """Return a conceptual polymer gear pump diagram."""
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 420"
width="760" height="420"
role="img" aria-label="Conceptual polymer gear pump with inlet, outlet, and gears">
<rect width="760" height="420" rx="24" fill="#101827"/>
<text x="42" y="55" fill="#f8fafc" font-family="Arial,sans-serif"
font-size="27" font-weight="700">Polymer gear pump</text>
<text x="42" y="82" fill="#94a3b8" font-family="Arial,sans-serif"
font-size="14">Conceptual flow diagram</text>
<rect x="218" y="121" width="328" height="216" rx="35"
fill="#243448" stroke="#7dd3fc" stroke-width="5"/>
<rect x="75" y="204" width="145" height="50" rx="10"
fill="#334155" stroke="#7dd3fc" stroke-width="4"/>
<rect x="544" y="204" width="142" height="50" rx="10"
fill="#334155" stroke="#7dd3fc" stroke-width="4"/>
<circle cx="326" cy="229" r="67" fill="#334155"
stroke="#fbbf24" stroke-width="10"/>
<circle cx="438" cy="229" r="67" fill="#334155"
stroke="#fbbf24" stroke-width="10"/>
<circle cx="326" cy="229" r="19" fill="#fbbf24"/>
<circle cx="438" cy="229" r="19" fill="#fbbf24"/>
<path d="M132 229h55m-14-12 14 12-14 12M577 229h70m-14-12 14 12-14 12"
fill="none" stroke="#38bdf8" stroke-width="7" stroke-linecap="round"
stroke-linejoin="round"/>
<text x="91" y="188" fill="#e2e8f0" font-family="Arial,sans-serif"
font-size="16">Inlet</text>
<text x="592" y="188" fill="#e2e8f0" font-family="Arial,sans-serif"
font-size="16">Outlet</text>
<text x="316" y="372" fill="#cbd5e1" font-family="Arial,sans-serif"
font-size="15">Intermeshing gears move polymer through the casing</text>
</svg>"""


def build_cutter_svg() -> str:
    """Return a conceptual pelletizer/strand cutter diagram."""
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 420"
width="760" height="420"
role="img" aria-label="Conceptual pellet cutter with rotor blades, bed knife, and strand feed">
<rect width="760" height="420" rx="24" fill="#101827"/>
<text x="42" y="55" fill="#f8fafc" font-family="Arial,sans-serif"
font-size="27" font-weight="700">Pellet cutter</text>
<text x="42" y="82" fill="#94a3b8" font-family="Arial,sans-serif"
font-size="14">Conceptual cutting-chamber diagram</text>
<rect x="150" y="120" width="130" height="46" rx="8"
fill="#334155" stroke="#7dd3fc" stroke-width="4"/>
<text x="163" y="150" fill="#e2e8f0" font-family="Arial,sans-serif"
font-size="14">Strand feed</text>
<path d="M280 143h60m-14-12 14 12-14 12" fill="none" stroke="#38bdf8"
stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
<rect x="220" y="170" width="340" height="180" rx="24"
fill="#243448" stroke="#7dd3fc" stroke-width="5"/>
<line x1="220" y1="260" x2="560" y2="260" stroke="#94a3b8" stroke-width="6"/>
<text x="390" y="248" fill="#e2e8f0" font-family="Arial,sans-serif"
font-size="14" text-anchor="middle">Bed knife</text>
<circle cx="390" cy="260" r="70" fill="none" stroke="#fbbf24" stroke-width="6"/>
<circle cx="390" cy="260" r="10" fill="#fbbf24"/>
<g stroke="#fbbf24" stroke-width="6" stroke-linecap="round">
<line x1="390" y1="260" x2="390" y2="192"/>
<line x1="390" y1="260" x2="450" y2="295"/>
<line x1="390" y1="260" x2="330" y2="295"/>
</g>
<text x="390" y="395" fill="#cbd5e1" font-family="Arial,sans-serif"
font-size="15" text-anchor="middle">Rotating blades shear strands against the bed knife into pellets</text>
<rect x="600" y="235" width="120" height="50" rx="10"
fill="#334155" stroke="#7dd3fc" stroke-width="4"/>
<text x="617" y="265" fill="#e2e8f0" font-family="Arial,sans-serif"
font-size="14">Pellets out</text>
<path d="M560 260h34m-14-12 14 12-14 12" fill="none" stroke="#38bdf8"
stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""
