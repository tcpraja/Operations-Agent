import ast
import logging
import math
import operator
import re
from concurrent.futures import ThreadPoolExecutor

import httpx
from agents import function_tool

from app.config import settings


logger = logging.getLogger(__name__)


# =========================================================
# SAFE ARITHMETIC / ENGINEERING CALCULATOR
# =========================================================

_ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
}

_ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def safe_calculate(expression: str) -> float:
    """
    Evaluate a simple arithmetic/engineering expression
    (numbers, + - * / // % **, parentheses, and a small
    whitelist of math functions/constants) without using
    Python's eval, so it can never execute arbitrary code.
    """

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):

        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed.")

        if isinstance(node, ast.BinOp):
            operator_fn = _ALLOWED_BINARY_OPERATORS.get(type(node.op))
            if operator_fn is None:
                raise ValueError("Unsupported operator.")
            return operator_fn(evaluate(node.left), evaluate(node.right))

        if isinstance(node, ast.UnaryOp):
            operator_fn = _ALLOWED_UNARY_OPERATORS.get(type(node.op))
            if operator_fn is None:
                raise ValueError("Unsupported operator.")
            return operator_fn(evaluate(node.operand))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
                raise ValueError("Unsupported function.")
            args = [evaluate(arg) for arg in node.args]
            return _ALLOWED_FUNCTIONS[node.func.id](*args)

        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_CONSTANTS:
                return _ALLOWED_CONSTANTS[node.id]
            raise ValueError("Unsupported name.")

        raise ValueError("Unsupported expression.")

    return evaluate(tree)


# =========================================================
# PURE PYTHON BUSINESS LOGIC
# Easy to test without an LLM
# =========================================================

def calculate_loss(
    downtime_minutes: float,
    loss_rate_usd_per_hour: float,
) -> dict:

    downtime_hours = downtime_minutes / 60

    loss = downtime_hours * loss_rate_usd_per_hour

    return {
        "downtime_hours": round(downtime_hours, 2),
        "estimated_loss_usd": round(loss, 2),
    }


def get_playbook(equipment: str) -> dict:

    playbooks = {
        "cutter": [
            "Inspect cutter blade condition",
            "Check cutter bearing temperature",
            "Check vibration",
            "Check cutter alignment",
            "Check motor current",
            "Review recent trip alarms",
        ],
        "pump": [
            "Check suction pressure",
            "Check discharge pressure",
            "Inspect mechanical seal",
            "Check motor current",
            "Check vibration",
        ],
        "compressor": [
            "Check discharge temperature",
            "Check suction pressure",
            "Check lubrication system",
            "Check vibration",
            "Review trip alarms",
        ],
    }

    equipment_lower = equipment.lower()

    for key, checks in playbooks.items():

        if key in equipment_lower:

            return {
                "equipment": key,
                "checks": checks,
            }

    return {
        "equipment": equipment,
        "checks": [
            "Inspect equipment condition",
            "Review alarm history",
            "Review maintenance history",
            "Check process parameters",
        ],
    }


# =========================================================
# AGENT TOOLS
# These wrappers expose the Python functions to the LLM
# =========================================================

@function_tool
def calculate_production_loss(
    downtime_minutes: float,
    loss_rate_usd_per_hour: float,
) -> dict:
    """Calculate production loss caused by equipment downtime."""

    print("\n[TOOL CALLED] calculate_production_loss")

    result = calculate_loss(
        downtime_minutes,
        loss_rate_usd_per_hour,
    )

    print(f"[TOOL RESULT] {result}\n")

    return result


@function_tool
def get_equipment_playbook(
    equipment: str,
) -> dict:
    """Get standard troubleshooting checks for manufacturing equipment."""

    print("\n[TOOL CALLED] get_equipment_playbook")
    print(f"Equipment requested: {equipment}")

    result = get_playbook(equipment)

    print(f"[TOOL RESULT] {result}\n")

    return result


# =========================================================
# WEB SEARCH
# =========================================================

def _search_web_firecrawl(
    query: str,
    max_results: int,
) -> list[dict]:
    """Use Firecrawl's web search when configured — far more
    relevant than unofficial DuckDuckGo scraping for specific
    technical/commercial queries (spare parts, suppliers, model
    numbers), and doesn't choke on special characters like a
    literal '$' the way a scraped search box can."""

    if not settings.FIRECRAWL_API_KEY:
        return []

    try:
        response = httpx.post(
            f"{FIRECRAWL_BASE_URL}/search",
            headers={
                "Authorization": f"Bearer {settings.FIRECRAWL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "limit": max_results,
            },
            timeout=15,
        )
        response.raise_for_status()
        results = response.json().get("data", [])

    except Exception:
        logger.warning(
            "firecrawl_text_search_failed query=%s",
            query,
        )
        return []

    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
        }
        for item in results
        if item.get("url")
    ]


def search_web(
    query: str,
    max_results: int = 5,
) -> list[dict]:
    """
    Search the open web and return lightweight result summaries.
    Tries Firecrawl first (when configured), since its search
    relevance is far better than unofficial DuckDuckGo scraping;
    falls back to DuckDuckGo otherwise.

    Failures (network issues, rate limiting, etc.) are
    swallowed and reported as an empty result list so a
    web outage never breaks incident analysis.
    """

    firecrawl_results = _search_web_firecrawl(query, max_results)
    if firecrawl_results:
        return firecrawl_results

    import time

    from ddgs import DDGS

    # A short per-round timeout bounds worst-case latency:
    # the "auto" backend fans out across several search
    # engines and some of them (Google, Brave, etc.)
    # frequently rate-limit or hang, which without a tight
    # timeout can push a single search past a minute. Try
    # the fast single-engine backend first, and only fall
    # back to "auto" (still capped) if that finds nothing.
    # Backends are flaky (transient TLS/connection drops),
    # so the whole two-backend pass is retried once after a
    # short pause before giving up.
    for pass_number in range(2):

        for backend in (
            "duckduckgo",
            "auto",
        ):

            try:

                raw_results = DDGS(
                    timeout=4
                ).text(
                    query,
                    backend=backend,
                    max_results=max_results,
                )

                results = [
                    {
                        "title": item.get(
                            "title", ""
                        ),
                        "url": item.get(
                            "href", ""
                        ),
                        "snippet": item.get(
                            "body", ""
                        ),
                    }
                    for item in raw_results
                ]

                if results:
                    return results

            except Exception:

                logger.warning(
                    "web_search_backend_failed "
                    "backend=%s query=%s",
                    backend,
                    query,
                )

                continue

        if pass_number == 0:
            time.sleep(1.5)

    logger.warning(
        "web_search_failed query=%s",
        query,
    )

    return []


_IMAGE_QUERY_STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or",
    "real", "actual", "photo", "photos", "photograph",
    "picture", "pictures", "image", "images",
    "industrial", "equipment", "plant", "vessel", "process",
    "need", "want", "give", "show", "find", "get", "some",
    "any", "please", "web", "internet", "one", "me",
}


def _significant_query_terms(query: str) -> set[str]:
    """Content words from a search query, ignoring generic
    filler like 'industrial equipment' that every query shares
    and so can't be used to judge relevance."""

    words = re.findall(r"[a-z0-9]+", query.lower())
    return {
        word for word in words
        if len(word) > 2 and word not in _IMAGE_QUERY_STOPWORDS
    }


def _is_relevant_image(item: dict, terms: set[str]) -> bool:
    """Reject results whose title/source page never mentions any
    content word from the query. Unofficial image-search scraping
    sometimes returns confidently wrong results for uncommon
    technical terms (e.g. an acronym); a title/URL that shares
    nothing with the query is a strong signal of that, and
    showing a clearly wrong photo is worse than showing none."""

    if not terms:
        # A query with no real content word (everything filtered
        # out as generic filler) can't be judged for relevance —
        # treat it as unsearchable rather than accepting anything.
        return False

    haystack = f"{item.get('title', '')} {item.get('source_url', '')}".lower()

    return any(term in haystack for term in terms)


FIRECRAWL_BASE_URL = "https://api.firecrawl.dev/v1"


def _firecrawl_scrape_og_image(url: str, headers: dict) -> str | None:
    """Fetch one page's Open Graph image, the closest thing to a
    canonical representative photo a page declares about itself."""

    try:
        response = httpx.post(
            f"{FIRECRAWL_BASE_URL}/scrape",
            headers=headers,
            json={
                "url": url,
                "formats": ["markdown"],
                "onlyMainContent": True,
            },
            timeout=15,
        )
        response.raise_for_status()
        metadata = response.json().get("data", {}).get("metadata", {})
        return metadata.get("ogImage") or metadata.get("og:image")

    except Exception:
        return None


def _search_web_images_firecrawl(
    query: str,
    max_results: int,
) -> list[dict]:
    """Use Firecrawl's web search (much higher relevance than
    unofficial DuckDuckGo scraping for uncommon technical terms)
    to find real source pages, then scrape each page's declared
    Open Graph image as its representative photo."""

    if not settings.FIRECRAWL_API_KEY:
        return []

    headers = {
        "Authorization": f"Bearer {settings.FIRECRAWL_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = httpx.post(
            f"{FIRECRAWL_BASE_URL}/search",
            headers=headers,
            json={
                "query": query,
                "limit": max_results * 2,
            },
            timeout=15,
        )
        response.raise_for_status()
        results = response.json().get("data", [])

    except Exception:
        logger.warning(
            "firecrawl_search_failed query=%s",
            query,
        )
        return []

    if not results:
        return []

    with ThreadPoolExecutor(max_workers=4) as executor:
        og_images = list(
            executor.map(
                lambda result: _firecrawl_scrape_og_image(
                    result.get("url", ""), headers
                ),
                results,
            )
        )

    images = [
        {
            "title": result.get("title", ""),
            "image_url": og_image,
            "thumbnail_url": og_image,
            "source_url": result.get("url", ""),
        }
        for result, og_image in zip(results, og_images)
        if og_image
    ]

    return images[:max_results]


def search_web_images(
    query: str,
    max_results: int = 4,
) -> list[dict]:
    """
    Search the open web for real photos. Tries Firecrawl first
    (when configured) since its search relevance is far better
    for uncommon technical terms; falls back to unofficial
    DuckDuckGo/Bing image scraping otherwise. Returns lightweight
    references (thumbnail, full image URL, source page) — never
    the image bytes themselves, since these are third-party
    copyrighted images.

    Failures are swallowed and reported as an empty list so
    a web outage never breaks the chat response.
    """

    terms = _significant_query_terms(query)

    firecrawl_images = _search_web_images_firecrawl(query, max_results)
    relevant_firecrawl_images = [
        image for image in firecrawl_images
        if _is_relevant_image(image, terms)
    ]
    if relevant_firecrawl_images:
        return relevant_firecrawl_images

    from ddgs import DDGS

    for backend in (
        "duckduckgo",
        "bing",
        "auto",
    ):

        try:

            raw_results = DDGS(
                timeout=6
            ).images(
                query,
                backend=backend,
                max_results=max_results * 3,
            )

            candidates = [
                {
                    "title": item.get("title", ""),
                    "image_url": item.get("image", ""),
                    "thumbnail_url": item.get("thumbnail", ""),
                    "source_url": item.get("url", ""),
                }
                for item in raw_results
                if item.get("image")
            ]

            relevant = [
                candidate for candidate in candidates
                if _is_relevant_image(candidate, terms)
            ]

            if relevant:
                return relevant[:max_results]

            logger.warning(
                "web_image_search_no_relevant_results "
                "backend=%s query=%s",
                backend,
                query,
            )

        except Exception:

            logger.warning(
                "web_image_search_backend_failed "
                "backend=%s query=%s",
                backend,
                query,
            )

        continue

    logger.warning(
        "web_image_search_failed query=%s",
        query,
    )

    return []