import asyncio
import base64
import logging
import re
import time
import uuid
from types import SimpleNamespace

import httpx

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)

from agents import Runner

from app.agent import openrouter_client, operations_agent
from app.charts import (
    build_cutter_knowledge_pareto,
    build_custom_chart,
    build_failure_count_chart,
    build_failure_trend_chart,
    build_pareto_analysis,
    build_severity_dashboard,
)
from app.config import settings
from app.documents import (
    SUPPORTED_UPLOAD_EXTENSIONS,
    REPORT_MIME_TYPES,
    apply_edit,
    build_document,
    build_report,
    extract_text,
)
from app.database import (
    delete_all_conversations,
    delete_conversation,
    delete_conversations,
    get_conversation,
    get_incident_by_id,
    get_incidents,
    initialize_database,
    list_conversations,
    replace_conversation,
    save_chat_message,
    save_incident,
    set_conversation_title,
)
from app.guardrails import apply_safety_guardrails
from app.logging_config import setup_logging
from app.models import (
    ChatDecision,
    ChatRequest,
    ChatResponse,
    DeleteConversationsRequest,
    ExportRequest,
    FileDownload,
    IncidentActionPlan,
    IncidentAnalysisResponse,
    IncidentRequest,
    RenameConversationRequest,
    ReplaceConversationRequest,
    UploadResponse,
)
from app.rag import retrieve_knowledge
from app.security import verify_api_key
from app.tools import safe_calculate, search_web, search_web_images
from app.visuals import build_cutter_svg, build_fishbone_svg, build_pump_svg


# =========================================================
# LOGGING
# =========================================================

setup_logging()

logger = logging.getLogger(
    "operations-agent"
)


# =========================================================
# CUSTOM EXCEPTIONS
# =========================================================

class AgentProviderError(Exception):
    """Raised when the LLM/provider fails."""


class AgentExecutionError(Exception):
    """Raised when the agent fails unexpectedly."""


EQUIPMENT_PATTERN = re.compile(
    r"\b(?:pump|filter|motor|cutter|conveyor|compressor|"
    r"valve|extruder|heater|dryer|fan|gearbox|boiler|"
    r"chiller|reactor|mixer|turbine|furnace|press)s?\b",
    re.IGNORECASE,
)

DIAGRAM_REQUEST_PATTERN = re.compile(
    r"\b(?:imag\w*|pictur\w*|diagram\w*|drawing|sketch\w*|visual\w*|"
    r"schematic\w*|photo\w*)\b",
    re.IGNORECASE,
)

REAL_IMAGE_PATTERN = re.compile(
    r"\b(?:real|actual|photo\w*|internet|web)\b",
    re.IGNORECASE,
)

CUTTER_PATTERN = re.compile(r"\bcutter\w*\b", re.IGNORECASE)
PUMP_PATTERN = re.compile(r"\bpump\w*\b", re.IGNORECASE)

COST_PATTERN = re.compile(
    r"\b(?:cost\w*|price\w*|pricing|how much|expense\w*|"
    r"budget\w*|quote\w*)\b",
    re.IGNORECASE,
)

ONLY_PRICE_PATTERN = re.compile(
    r"\b(?:only|just)\b.{0,15}\bprice\w*\b|"
    r"\bprice\w*\b.{0,15}\b(?:only|just)\b",
    re.IGNORECASE,
)

PRICE_PATTERN = re.compile(
    r"(?:USD|US\$|\$|€|£|₹|INR|EUR|GBP)\s?\d[\d,]*(?:\.\d+)?"
    r"(?:\s?(?:-|–|to)\s?(?:USD|US\$|\$|€|£|₹|INR|EUR|GBP)?"
    r"\s?\d[\d,]*(?:\.\d+)?)?",
    re.IGNORECASE,
)

FISHBONE_PATTERN = re.compile(
    r"\bfishbone\b|\bishikawa\b|\bcause[- ]and[- ]effect\s+diagram\b",
    re.IGNORECASE,
)

PURE_ARITHMETIC_PATTERN = re.compile(
    r"^[\d\s\+\-\*\/\.\(\)\^%]+$"
)

SYMPTOM_PATTERN = re.compile(
    r"\b(?:not running|not operating|not starting|won't start|"
    r"won't run|stopped|tripped|trip|leak\w*|noise|"
    r"vibrat\w*|pressure|overheat\w*|jam\w*|alarm|"
    r"fault|failure|failed|malfunction|stuck|seized|"
    r"low flow|no flow|high temperature|broken|down)\b",
    re.IGNORECASE,
)


def has_core_incident_details(incident: str) -> bool:
    """Recognize equipment and a symptom across user turns."""
    return bool(
        EQUIPMENT_PATTERN.search(incident)
        and SYMPTOM_PATTERN.search(incident)
    )


def missing_core_question(incident: str) -> str | None:
    """Ask for facts needed before generating an action plan."""
    has_equipment = bool(EQUIPMENT_PATTERN.search(incident))
    has_symptom = bool(SYMPTOM_PATTERN.search(incident))
    if not has_equipment and not has_symptom:
        return "Which equipment is involved, and what is happening?"
    if not has_equipment:
        return "Which equipment is affected?"
    if not has_symptom:
        return "What issue or symptom is the equipment showing?"
    return None


def equipment_search_term(incident: str) -> str:
    """Prefer the latest equipment detail for web queries."""
    for line in reversed(incident.splitlines()):
        if EQUIPMENT_PATTERN.search(line):
            return line.strip()
    return " ".join(incident.split())


async def run_ollama_native_json_fallback(prompt: str):
    """Use Ollama's schema mode when tool-driven JSON is malformed."""

    async with httpx.AsyncClient(
        timeout=settings.AGENT_TIMEOUT_SECONDS
    ) as client:
        for attempt in range(2):
            instructions = operations_agent.instructions
            if attempt:
                instructions += (
                    "\nKeep the answer brief. Use at most two items "
                    "per list and complete every JSON field."
                )

            response = await client.post(
                settings.AI_BASE_URL.removesuffix("/v1")
                + "/api/chat",
                json={
                    "model": settings.AI_MODEL,
                    "messages": [
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": prompt},
                    ],
                    "format": IncidentActionPlan.model_json_schema(),
                    "think": False,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": 4096,
                    },
                },
            )
            response.raise_for_status()
            payload = response.json()
            logger.info(
                "ollama_native_json_fallback_response "
                "attempt=%s done_reason=%s eval_count=%s",
                attempt + 1,
                payload.get("done_reason"),
                payload.get("eval_count"),
            )
            try:
                plan = IncidentActionPlan.model_validate_json(
                    payload["message"]["content"]
                )
                return SimpleNamespace(final_output=plan)
            except Exception as error:
                logger.warning(
                    "ollama_native_json_fallback_invalid "
                    "attempt=%s error_type=%s",
                    attempt + 1,
                    type(error).__name__,
                )
                if attempt:
                    raise


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Agentic AI service for manufacturing "
        "incident analysis."
    ),
    version=settings.APP_VERSION,
)


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

initialize_database()


# =========================================================
# REQUEST LOGGING MIDDLEWARE
# =========================================================

@app.middleware("http")
async def request_logging(
    request: Request,
    call_next,
):

    request_id = str(
        uuid.uuid4()
    )[:8]

    start_time = time.time()

    logger.info(
        f"request_started "
        f"request_id={request_id} "
        f"method={request.method} "
        f"path={request.url.path}"
    )

    try:

        response = await call_next(
            request
        )

    except Exception:

        logger.exception(
            f"request_failed "
            f"request_id={request_id}"
        )

        raise

    duration_ms = round(
        (
            time.time()
            - start_time
        )
        * 1000,
        2,
    )

    logger.info(
        f"request_completed "
        f"request_id={request_id} "
        f"status={response.status_code} "
        f"duration_ms={duration_ms}"
    )

    response.headers[
        "X-Request-ID"
    ] = request_id

    return response


# =========================================================
# AGENT ERROR CLASSIFICATION
# =========================================================

def classify_agent_error(
    error: Exception,
) -> Exception:

    error_name = type(
        error
    ).__name__.lower()

    error_message = str(
        error
    ).lower()

    provider_keywords = [
        "ratelimit",
        "rate limit",
        "openai",
        "openrouter",
        "api error",
        "connection",
        "service unavailable",
        "bad gateway",
        "modelbehaviorerror",
        "invalid json",
    ]

    if any(
        keyword in error_name
        or keyword in error_message
        for keyword in provider_keywords
    ):

        return AgentProviderError(
            str(error)
        )

    return AgentExecutionError(
        str(error)
    )


# =========================================================
# AGENT RETRY
# =========================================================

async def run_agent_with_retry(
    prompt: str,
    max_attempts: int = settings.MAX_AGENT_ATTEMPTS,
):

    last_error = None

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        try:

            logger.info(
                f"agent_attempt_started "
                f"attempt={attempt}"
            )

            result = await asyncio.wait_for(
                Runner.run(
                    operations_agent,
                    prompt,
                    max_turns=6,
                ),
                timeout=(
                    settings.AGENT_TIMEOUT_SECONDS
                ),
            )

            logger.info(
                f"agent_attempt_success "
                f"attempt={attempt}"
            )

            return result

        except asyncio.TimeoutError as error:

            last_error = error

            logger.warning(
                f"agent_attempt_timeout "
                f"attempt={attempt} "
                f"timeout_seconds="
                f"{settings.AGENT_TIMEOUT_SECONDS}"
            )

        except Exception as error:

            classified_error = (
                classify_agent_error(
                    error
                )
            )

            last_error = classified_error

            logger.warning(
                f"agent_attempt_failed "
                f"attempt={attempt} "
                f"error_type="
                f"{type(classified_error).__name__}"
            )

        if attempt < max_attempts:

            logger.info(
                f"agent_retrying "
                f"next_attempt="
                f"{attempt + 1}"
            )

            await asyncio.sleep(
                settings.RETRY_DELAY_SECONDS
            )

    if (
        settings.AI_PROVIDER == "ollama"
        and isinstance(last_error, AgentProviderError)
        and "invalid json" in str(last_error).lower()
    ):
        logger.info("ollama_native_json_fallback_started")
        try:
            result = await run_ollama_native_json_fallback(prompt)
            logger.info("ollama_native_json_fallback_success")
            return result
        except Exception as error:
            logger.warning(
                "ollama_native_json_fallback_failed "
                "error_type=%s",
                type(error).__name__,
            )

    raise last_error


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "service": (
            "manufacturing-operations-agent"
        ),
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "ai_provider": settings.AI_PROVIDER,
        "ai_model": settings.AI_MODEL,
    }


async def _upload_core(
    file: UploadFile,
    instruction: str | None,
) -> UploadResponse:
    """Read or edit an uploaded PPTX, DOCX, XLSX, PDF, JPG, or PNG file."""
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. Supported: "
                + ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
            ),
        )

    data = await file.read()
    cleaned_instruction = (instruction or "").strip()

    edit_intent = bool(
        cleaned_instruction
        and re.search(r"\b(add|append|insert|replace|change|update|edit)\b", cleaned_instruction, re.IGNORECASE)
    )
    read_intent = bool(
        cleaned_instruction
        and re.search(
            r"\b(describe|summarize|summarise|explain|analyze|analyse|"
            r"read|review|what|show|list|tell me|extract)\b",
            cleaned_instruction,
            re.IGNORECASE,
        )
    )

    try:
        if edit_intent and suffix not in (".txt", ".csv", ".jpg", ".jpeg", ".png"):
            edited_bytes = apply_edit(file.filename, data, cleaned_instruction)
            base_name = file.filename.rsplit(".", 1)[0]
            return UploadResponse(
                reply=f"I updated {file.filename} as requested. Download the edited file below.",
                download=FileDownload(
                    filename=f"{base_name}_edited{suffix}",
                    mime_type=REPORT_MIME_TYPES.get(suffix.lstrip("."), "application/octet-stream"),
                    data_base64=base64.b64encode(edited_bytes).decode("ascii"),
                ),
            )

        extracted = extract_text(file.filename, data)
        is_image = suffix in (".jpg", ".jpeg", ".png")
        if not extracted.strip():
            if is_image:
                return UploadResponse(
                    reply=(
                        f"I scanned {file.filename} for text (e.g. a nameplate or "
                        "alarm screen) but found none. I can't yet visually inspect "
                        "equipment photos for damage, leaks, or wear — describe what "
                        "you see (equipment type, symptom, any visible defect) and "
                        "I'll help you troubleshoot it."
                    )
                )
            return UploadResponse(reply=f"I could not find readable text in {file.filename}.")

        preview = extracted[:4000]
        if is_image:
            reply = (
                f"Here is the text I found in {file.filename} "
                f"({len(extracted)} characters):\n\n{preview}"
            )
        else:
            reply = (
                f"I read {file.filename} ({len(extracted)} characters). "
                f"Here is what it contains:\n\n{preview}"
            )
        if cleaned_instruction and not is_image and not read_intent:
            reply = (
                f"I read {file.filename} but couldn't tell what change to make from "
                f"\"{cleaned_instruction}\". Try phrasing it like "
                "'replace \"old text\" with \"new text\"' or 'add ...'. "
                f"Here is the extracted content in the meantime:\n\n{preview}"
            )
        return UploadResponse(reply=reply, extracted_text=extracted)

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception:
        logger.exception("upload_processing_failed filename=%s", file.filename)
        raise HTTPException(status_code=500, detail="Could not process the uploaded file.")


@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    instruction: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
    authorized: bool = Depends(verify_api_key),
):
    """Read or edit an uploaded file, then persist the turn so it
    survives a frontend refresh or a new browser session."""

    filename = file.filename
    response = await _upload_core(file, instruction)

    if session_id:
        try:
            save_chat_message(
                session_id, "user", "text",
                f"Uploaded {filename}" + (f" — {instruction.strip()}" if instruction else ""),
            )
            save_chat_message(session_id, "assistant", "upload", response.model_dump())
        except Exception:
            logger.exception("upload_message_persist_failed session_id=%s", session_id)

    return response


@app.post("/export")
async def export_report(
    request: ExportRequest,
    authorized: bool = Depends(verify_api_key),
):
    """Render an incident analysis as a downloadable PPTX, DOCX, XLSX, or PDF report."""
    try:
        report_bytes = build_report(request.analysis, request.format)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception:
        logger.exception("export_report_failed format=%s", request.format)
        raise HTTPException(status_code=500, detail="Could not build the report.")

    return FileDownload(
        filename=f"incident_report.{request.format}",
        mime_type=REPORT_MIME_TYPES[request.format],
        data_base64=base64.b64encode(report_bytes).decode("ascii"),
    )


async def _chat_core(
    request: ChatRequest,
    authorized: bool = True,
) -> ChatResponse:
    """Let the model choose a reply or an available tool."""
    latest = request.message.strip()
    prior_user_messages = [
        turn.content for turn in request.history[-8:] if turn.role == "user"
    ]
    knowledge_query = " ".join(prior_user_messages[-2:] + [latest])
    try:
        knowledge_matches = retrieve_knowledge(knowledge_query, max_results=2)
    except Exception:
        logger.exception("chat_knowledge_retrieval_failed")
        knowledge_matches = []
    knowledge_context = "\n".join(
        f"[{match['source']}] {match['content'][:700]}"
        for match in knowledge_matches
    )
    messages = [{
        "role": "system",
        "content": (
            "You are a Manufacturing Operations AI Agent: a specialized "
            "assistant for plant engineers, operators, supervisors, "
            "maintenance teams, and operations managers. Your scope is "
            "manufacturing equipment, production problems, incidents, "
            "downtime, reliability, maintenance, root cause analysis, "
            "OEE, quality, safety, process parameters, and simple "
            "equipment diagrams. Do not behave as a general-purpose "
            "assistant on topics unrelated to manufacturing operations; "
            "for those, briefly say the request is outside this agent's "
            "manufacturing-operations scope. "
            "You have tools for incident analysis, a conceptual pump "
            "diagram, a conceptual cutter (pelletizer) diagram, a real "
            "photo search of the web, a live web price/cost lookup, a "
            "Fishbone (Ishikawa) root-cause diagram, and a Pareto chart "
            "from local cutter benchmarking records. "
            "Choose one action from reply, analyze, pump_diagram, "
            "cutter_diagram, web_image, web_cost, fishbone_diagram, "
            "calculate, custom_chart, generate_document, knowledge_pareto. "
            "Use web_image when the user explicitly asks for a real, "
            "actual, or photo/picture from the internet or the web, "
            "rather than a conceptual diagram. Web images are "
            "third-party results, not verified plant photos. Do NOT use "
            "web_image for a request about a video, YouTube, or a video "
            "link — no tool can fetch or verify a specific video. For "
            "those, use reply and say plainly that you cannot provide or "
            "verify a specific video link. When you choose web_image, set "
            "web_image_query to a specific, unambiguous English search "
            "phrase for the equipment plus 'industrial equipment' — "
            "always expand an abbreviation or acronym to its full name "
            "first (e.g. 'CSTR' becomes 'continuous stirred-tank reactor "
            "industrial equipment'), since searching the bare acronym "
            "returns unrelated results. "
            "Use web_cost whenever the user asks about the cost, price, "
            "or a quote for equipment or spare parts (including a "
            "follow-up that only names a part after a cost question was "
            "already asked, e.g. 'mechanical seal' right after 'cost of "
            "pump seal'). Never answer a cost/price question yourself "
            "from memory or say you lack access to pricing — always use "
            "web_cost so it can search the live web instead. "
            "Use fishbone_diagram whenever the user asks for a Fishbone, "
            "Ishikawa, or cause-and-effect diagram for a problem, or asks "
            "you to turn a root-cause analysis into a diagram/image. When "
            "you choose fishbone_diagram, set fishbone_problem to a short "
            "statement of the problem, and fill whichever of "
            "fishbone_machine, fishbone_method, fishbone_material, "
            "fishbone_manpower, fishbone_measurement, fishbone_environment "
            "are relevant with 1-2 short concrete causes each (leave "
            "irrelevant categories as empty lists); do not repeat that "
            "content in reply since the diagram will show it — keep reply "
            "to one short sentence. "
            "Use calculate for arithmetic, engineering, or basic "
            "chemistry calculations (e.g. unit conversions, "
            "concentration, molar mass ratios, production-loss style "
            "arithmetic). Never compute the number yourself in reply — "
            "set calculation_expression to a plain arithmetic expression "
            "using + - * / ** () and the functions sqrt/log/log10/exp/"
            "sin/cos/tan/abs/round with constants pi and e, and the tool "
            "will evaluate it exactly. "
            "Use custom_chart when the user wants a trend/line chart or "
            "pie chart of specific data points. Set chart_type to "
            "'trend' or 'pie', chart_title, and matching chart_labels / "
            "chart_values lists using only numbers actually given in the "
            "conversation — never invent data points. "
            "You CAN generate real downloadable Word, PowerPoint, Excel, "
            "and PDF files — never say you cannot create or export one. "
            "Use generate_document whenever the user asks you to develop, "
            "create, build, write, or deliver a document, report, slide "
            "deck, presentation, ppt, doc, or Excel/spreadsheet template "
            "on some topic (as opposed to exporting an incident analysis "
            "already shown in this conversation, which the UI already "
            "offers as a download button). Set document_format to docx, "
            "pptx, xlsx, or pdf, document_title to a short title, and "
            "document_section_headings to a list of section/slide/sheet "
            "titles; document_section_bullets must be the SAME LENGTH "
            "list, one string per heading, with that section's bullet "
            "points joined by ' | ' (pipe with spaces). Use your own "
            "manufacturing/engineering knowledge to write real, useful "
            "content — do not leave sections empty. Keep reply to one "
            "short sentence since the file carries the content. "
            "Use knowledge_pareto when the user asks for a cutter failure "
            "Pareto based on past data, including a follow-up request. "
            "Local knowledge is available; never claim it is inaccessible. "
            "The benchmarking counts are sites reporting failure categories, "
            "not trip event frequencies. Cite the source when using excerpts. "
            "Use reply for greetings, corrections, refusals such as "
            "'no', ordinary questions, equipment explanations, and "
            "incomplete incident details. When explaining a piece of "
            "equipment in reply, briefly cover what it does, main "
            "components, operating principle, common failure modes, and "
            "safety precautions, only as far as relevant to the question — "
            "keep it concise, not an essay. Answer naturally in reply; "
            "do not demand an incident. Ask only the minimum question "
            "needed when information is genuinely missing; never ask "
            "several questions at once. "
            "Use analyze when the user wants help with an equipment "
            "problem and the conversation identifies both equipment "
            "and symptom. 'Not running' is already a symptom; "
            "'not running' followed by 'pump' is enough to analyze. "
            "Do not ask for alarm codes, model numbers, or other "
            "optional details before analyzing. In incident, "
            "combine relevant facts from "
            "the USER's messages without inventing details. Never invent "
            "plant-specific temperatures, pressures, alarm values, or "
            "maintenance history; if unavailable, say data is required "
            "for confirmation. Safety always outranks production, cost, "
            "or throughput. "
            "Use pump_diagram when the user requests a visual of a pump, "
            "and cutter_diagram when the user requests a visual of a "
            "cutter or pelletizer. Never substitute one diagram for the "
            "other equipment type. Both diagrams are conceptual, not "
            "model-specific. "
            "The newest user message controls intent; earlier turns "
            "provide context. Keep reply brief and conversational. "
            "In polymer processing, a polymer gear pump moves "
            "polymer melt; its gears need not be polymer.\n"
            "Relevant local knowledge excerpts:\n" + (knowledge_context or "No matching excerpt.")
        ),
    }]
    messages.extend(
        {"role": turn.role, "content": turn.content}
        for turn in request.history[-8:]
    )
    messages.append({"role": "user", "content": request.message.strip()})

    decision = None
    last_error = None
    for attempt in range(1, settings.MAX_AGENT_ATTEMPTS + 1):
        try:
            if settings.AI_PROVIDER == "ollama":
                async with httpx.AsyncClient(
                    timeout=settings.AGENT_TIMEOUT_SECONDS
                ) as client:
                    response = await client.post(
                        settings.AI_BASE_URL.removesuffix("/v1")
                        + "/api/chat",
                        json={
                            "model": settings.AI_MODEL,
                            "messages": messages,
                            "format": ChatDecision.model_json_schema(),
                            "think": False,
                            "stream": False,
                            "options": {
                                "temperature": 0.2,
                                "num_predict": 1024,
                            },
                        },
                    )
                    response.raise_for_status()
                    decision = ChatDecision.model_validate_json(
                        response.json()["message"]["content"]
                    )
            else:
                response = await openrouter_client.chat.completions.create(
                    model=settings.AI_MODEL,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "chat_decision",
                            "schema": ChatDecision.model_json_schema(),
                        },
                    },
                    max_tokens=1024,
                )
                decision = ChatDecision.model_validate_json(
                    response.choices[0].message.content or "{}"
                )
            break
        except Exception as error:
            last_error = error
            logger.warning(
                f"chat_model_attempt_failed "
                f"attempt={attempt} "
                f"error={error}"
            )
            if attempt < settings.MAX_AGENT_ATTEMPTS:
                await asyncio.sleep(settings.RETRY_DELAY_SECONDS)

    if decision is None:
        logger.error(f"chat_model_failure error={last_error}")
        raise HTTPException(
            status_code=502,
            detail="Chat model temporarily unavailable.",
        )

    # The model may answer conversationally even after a user
    # supplies the missing equipment or symptom. In that case,
    # use the analysis tool with the facts from recent user turns.
    recent_user_facts = [
        turn.content
        for turn in request.history[-8:]
        if turn.role == "user"
        and (
            EQUIPMENT_PATTERN.search(turn.content)
            or SYMPTOM_PATTERN.search(turn.content)
        )
    ]
    if (
        EQUIPMENT_PATTERN.search(latest)
        or SYMPTOM_PATTERN.search(latest)
    ):
        recent_user_facts.append(latest)
    combined_facts = "\n".join(recent_user_facts)
    if (
        decision.action == "reply"
        and has_core_incident_details(combined_facts)
        and (
            EQUIPMENT_PATTERN.search(latest)
            or SYMPTOM_PATTERN.search(latest)
        )
        and not re.match(
            r"^(?:what|how|why|who|where|can|could|draw|"
            r"create|develop|show|make)\b",
            latest,
            re.IGNORECASE,
        )
    ):
        decision = ChatDecision(
            action="analyze",
            reply="Here is the incident analysis.",
            incident=combined_facts,
        )
    elif (
        decision.action == "analyze"
        and has_core_incident_details(combined_facts)
        and not has_core_incident_details(decision.incident or "")
    ):
        decision.incident = combined_facts

    if (
        DIAGRAM_REQUEST_PATTERN.search(latest)
        and not FISHBONE_PATTERN.search(latest)
    ):
        if REAL_IMAGE_PATTERN.search(latest):
            if decision.action != "web_image":
                decision = ChatDecision(action="web_image", reply="")
        elif CUTTER_PATTERN.search(latest) and not PUMP_PATTERN.search(latest):
            if decision.action != "cutter_diagram":
                decision = ChatDecision(action="cutter_diagram", reply="Here is a cutter diagram.")
        elif PUMP_PATTERN.search(latest) and not CUTTER_PATTERN.search(latest):
            if decision.action != "pump_diagram":
                decision = ChatDecision(action="pump_diagram", reply="Here is a pump diagram.")

    cost_context_text = " ".join(prior_user_messages[-3:] + [latest])
    if (
        COST_PATTERN.search(cost_context_text)
        and decision.action == "reply"
    ):
        decision = ChatDecision(action="web_cost", reply="")

    recent_assistant_text = " ".join(
        turn.content for turn in request.history[-4:] if turn.role == "assistant"
    ).lower()
    fishbone_followup = (
        ("fishbone" in recent_assistant_text or "ishikawa" in recent_assistant_text)
        and (
            FISHBONE_PATTERN.search(latest)
            or DIAGRAM_REQUEST_PATTERN.search(latest)
            or re.search(r"\bdevelop\w*\b", latest, re.IGNORECASE)
        )
    )
    if (
        (FISHBONE_PATTERN.search(latest) or fishbone_followup)
        and decision.action != "fishbone_diagram"
        and any([
            decision.fishbone_machine, decision.fishbone_method,
            decision.fishbone_material, decision.fishbone_manpower,
            decision.fishbone_measurement, decision.fishbone_environment,
        ])
    ):
        decision.action = "fishbone_diagram"

    if (
        PURE_ARITHMETIC_PATTERN.match(latest)
        and any(character.isdigit() for character in latest)
    ):
        decision = ChatDecision(action="calculate", reply="", calculation_expression=latest)

    pareto_context = " ".join(prior_user_messages[-3:] + [latest]).lower()
    if (
        "pareto" in pareto_context
        and "cutter" in pareto_context
        and ("pareto" in latest.lower() or re.search(r"\b(?:past|historical|previous|data|records)\b", latest, re.I))
    ):
        decision.action = "knowledge_pareto"

    if decision.action == "knowledge_pareto":
        pareto = build_cutter_knowledge_pareto()
        if pareto is None:
            return ChatResponse(reply="I found no structured cutter failure counts in the knowledge folder to plot.")
        return ChatResponse(
            reply=(
                f"Here is the cutter failure Pareto from {pareto['source']} "
                f"({pareto['sheet']} sheet). {pareto['note']}"
            ),
            pareto=pareto,
        )

    if decision.action == "calculate":
        expression = (decision.calculation_expression or latest).strip()
        try:
            result = safe_calculate(expression)
        except Exception:
            return ChatResponse(
                reply=(
                    "I couldn't safely evaluate that expression. Please "
                    "phrase it as plain arithmetic, e.g. '(45 * 1.8) / 3'."
                )
            )
        formatted_result = (
            f"{result:g}" if isinstance(result, float) else str(result)
        )
        return ChatResponse(reply=f"{expression} = {formatted_result}")

    if decision.action == "custom_chart":
        image = build_custom_chart(
            decision.chart_type or "trend",
            decision.chart_title or "Chart",
            decision.chart_labels,
            decision.chart_values,
        )
        if image is None:
            return ChatResponse(
                reply=(
                    decision.reply.strip()
                    or "Give me the labels and matching numeric values "
                    "you'd like charted (e.g. 'Jan: 12, Feb: 18, Mar: 9')."
                )
            )
        return ChatResponse(
            reply=decision.reply.strip() or "Here is the chart.",
            chart_image=image,
        )

    if decision.action == "generate_document":
        headings = decision.document_section_headings
        bullets = [
            [part.strip() for part in entry.split("|") if part.strip()]
            for entry in decision.document_section_bullets
        ]
        if (
            not decision.document_format
            or not headings
            or len(headings) != len(bullets)
            or not any(bullets)
        ):
            return ChatResponse(
                reply=(
                    decision.reply.strip()
                    or "Tell me the document format (Word, PowerPoint, "
                    "Excel, or PDF) and the topic, and I'll generate it "
                    "for you."
                )
            )
        try:
            file_bytes = build_document(
                decision.document_format,
                decision.document_title or "Document",
                headings,
                bullets,
            )
        except Exception:
            logger.exception("chat_generate_document_failed")
            return ChatResponse(reply="I couldn't build that document. Please try again.")

        base_name = re.sub(r"[^\w\- ]+", "", decision.document_title or "document").strip() or "document"
        return ChatResponse(
            reply=decision.reply.strip() or f"Here is your {decision.document_format.upper()} file.",
            download=FileDownload(
                filename=f"{base_name}.{decision.document_format}",
                mime_type=REPORT_MIME_TYPES[decision.document_format],
                data_base64=base64.b64encode(file_bytes).decode("ascii"),
            ),
        )

    if decision.action == "fishbone_diagram":
        categories = {
            "Machine": decision.fishbone_machine,
            "Method": decision.fishbone_method,
            "Material": decision.fishbone_material,
            "Manpower": decision.fishbone_manpower,
            "Measurement": decision.fishbone_measurement,
            "Environment": decision.fishbone_environment,
        }
        if not any(categories.values()):
            return ChatResponse(
                reply=(
                    decision.reply.strip()
                    or "Describe the problem and its likely contributing "
                    "factors first, and I'll turn that into a Fishbone "
                    "diagram."
                )
            )
        svg = build_fishbone_svg(
            decision.fishbone_problem or "Reported problem",
            categories,
        )
        return ChatResponse(
            reply=decision.reply.strip() or "Here is the Fishbone (Ishikawa) diagram.",
            image_svg=svg,
        )

    if decision.action == "web_cost":
        cost_query_terms = " ".join(prior_user_messages[-3:] + [latest])
        equipment_term = equipment_search_term(cost_query_terms)
        only_price_requested = bool(ONLY_PRICE_PATTERN.search(latest))
        results = await asyncio.to_thread(
            search_web,
            f'{equipment_term} spare parts price cost "$" buy quote',
        )
        if not results:
            return ChatResponse(
                reply=(
                    "I couldn't find pricing information on the web right "
                    "now. Cost varies by supplier, region, and equipment "
                    "model — request a formal quote from the manufacturer "
                    "or a distributor."
                )
            )

        priced_results = []
        for item in results[:6]:
            prices = PRICE_PATTERN.findall(
                f"{item.get('title', '')} {item.get('snippet', '')}"
            )
            if prices:
                priced_results.append((item, prices))

        web_sources = [
            {"title": item["title"], "url": item["url"]}
            for item in results
            if item.get("url")
        ]

        if priced_results:
            price_lines = "\n".join(
                f"- {', '.join(dict.fromkeys(prices))} — "
                f"[{item['title']}]({item['url']})"
                for item, prices in priced_results
            )
            if only_price_requested:
                reply = (
                    "Price figures found in web listings (third-party, "
                    f"not a formal quote):\n\n{price_lines}"
                )
            else:
                listing_lines = "\n".join(
                    f"- [{item['title']}]({item['url']}): {item['snippet'][:220]}"
                    for item in results[:5]
                    if item.get("url")
                )
                reply = (
                    "Here is what I found on the web about pricing (these "
                    "are third-party listings, not a formal quote — verify "
                    f"before purchasing):\n\n{listing_lines}\n\n"
                    f"**Price figures found:**\n{price_lines}"
                )
            return ChatResponse(reply=reply, web_sources=web_sources)

        listing_lines = "\n".join(
            f"- [{item['title']}]({item['url']})"
            for item in results[:5]
            if item.get("url")
        )
        return ChatResponse(
            reply=(
                "I searched the web but none of the result snippets "
                "contained an explicit price figure — the pages below may "
                "still have pricing once opened, or may require a request-a-"
                f"quote form:\n\n{listing_lines}"
            ),
            web_sources=web_sources,
        )

    if decision.action == "web_image" and re.search(r"\byoutube\b|\bvideo\w*\b", latest, re.IGNORECASE):
        return ChatResponse(
            reply=(
                "I can't fetch, verify, or link a specific video — no tool "
                "here has access to video search. I can share a real photo "
                "or a conceptual diagram instead if that helps."
            )
        )

    if decision.action in ("web_image", "pump_diagram", "cutter_diagram"):
        media_context = " ".join(prior_user_messages[-3:] + [latest])
        if not EQUIPMENT_PATTERN.search(media_context):
            return ChatResponse(
                reply="Which equipment are you asking about (e.g. pump, cutter, compressor, motor)?"
            )

    if decision.action == "web_image":
        # A follow-up like "need web photo" carries no equipment
        # word of its own — fall back to the recent conversation
        # context (already validated above), not just this message,
        # or the search loses the equipment entirely.
        equipment_term = equipment_search_term(media_context)
        if CUTTER_PATTERN.search(media_context):
            search_terms = "pellet cutter pelletizer industrial equipment"
        elif PUMP_PATTERN.search(media_context):
            search_terms = "polymer gear pump industrial equipment"
        elif decision.web_image_query and decision.web_image_query.strip():
            search_terms = decision.web_image_query.strip()
        else:
            search_terms = f"{equipment_term} industrial equipment"

        images = await asyncio.to_thread(search_web_images, search_terms)

        if not images:
            fallback_svg = (
                build_cutter_svg()
                if CUTTER_PATTERN.search(latest)
                else build_pump_svg()
            )
            return ChatResponse(
                reply=(
                    "I couldn't find a real photo for that on the web right "
                    "now. Here is a conceptual diagram instead."
                ),
                image_svg=fallback_svg,
            )

        return ChatResponse(
            reply=(
                decision.reply.strip()
                or "Here are real photos found on the web. These are "
                "third-party images — verify against your own equipment "
                "before relying on them."
            ),
            web_images=images,
        )

    if decision.action == "pump_diagram":
        return ChatResponse(
            reply=decision.reply.strip() or "Here is a pump diagram.",
            image_svg=build_pump_svg(),
        )

    if decision.action == "cutter_diagram":
        return ChatResponse(
            reply=decision.reply.strip() or "Here is a cutter diagram.",
            image_svg=build_cutter_svg(),
        )

    if decision.action == "analyze" and decision.incident:
        analysis = await analyze_incident(
            IncidentRequest(incident=decision.incident),
            authorized=True,
        )
        if analysis.needs_clarification:
            return ChatResponse(
                reply=analysis.clarification_question
                or "What equipment and symptom are involved?"
            )
        return ChatResponse(
            reply=decision.reply.strip() or "Here is the analysis.",
            analysis=analysis.model_dump(),
        )

    return ChatResponse(
        reply=decision.reply.strip() or "How can I help?"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    authorized: bool = Depends(verify_api_key),
):
    """Run the chat decision, then persist both sides of the
    turn so past conversations survive a frontend refresh or
    a new browser session."""

    response = await _chat_core(request, authorized)

    session_id = request.session_id
    if session_id:
        try:
            save_chat_message(session_id, "user", "text", request.message.strip())
            save_chat_message(session_id, "assistant", "chat_response", response.model_dump())
        except Exception:
            logger.exception("chat_message_persist_failed session_id=%s", session_id)

    return response


# =========================================================
# CONVERSATION HISTORY
# =========================================================

@app.get("/conversations")
async def conversations(
    limit: int = 30,
    authorized: bool = Depends(verify_api_key),
):
    """List recent conversations for the sidebar."""
    return {"conversations": list_conversations(limit=limit)}


@app.get("/conversations/{session_id}")
async def conversation(
    session_id: str,
    authorized: bool = Depends(verify_api_key),
):
    """Return every stored message for one conversation."""
    return {"messages": get_conversation(session_id)}


@app.put("/conversations/{session_id}/title")
async def rename_conversation(
    session_id: str,
    request: RenameConversationRequest,
    authorized: bool = Depends(verify_api_key),
):
    """Set a custom display title for a conversation."""
    set_conversation_title(session_id, request.title)
    return {"session_id": session_id, "title": request.title.strip()[:80]}


@app.put("/conversations/{session_id}")
async def replace_conversation_history(
    session_id: str,
    request: ReplaceConversationRequest,
    authorized: bool = Depends(verify_api_key),
):
    """Replace a conversation's entire stored history — used after
    the user edits an earlier message and regenerates the reply,
    so the discarded branch doesn't come back on a refresh."""
    replace_conversation(session_id, request.messages)
    return {"session_id": session_id, "message_count": len(request.messages)}


@app.delete("/conversations/{session_id}")
async def delete_one_conversation(
    session_id: str,
    authorized: bool = Depends(verify_api_key),
):
    """Delete a single conversation's chat history."""
    deleted_count = delete_conversation(session_id)
    return {"deleted_messages": deleted_count}


@app.delete("/conversations")
async def delete_many_conversations(
    request: DeleteConversationsRequest | None = None,
    delete_all: bool = False,
    authorized: bool = Depends(verify_api_key),
):
    """Delete several conversations, or every conversation
    when called with ?delete_all=true."""
    if delete_all:
        deleted_count = delete_all_conversations()
    else:
        session_ids = request.session_ids if request else []
        deleted_count = delete_conversations(session_ids)
    return {"deleted_messages": deleted_count}


# =========================================================
# INCIDENT ANALYSIS
# =========================================================

@app.post(
    "/analyze",
    response_model=IncidentAnalysisResponse,
)
async def analyze_incident(
    request: IncidentRequest,
    authorized: bool = Depends(
        verify_api_key
    ),
):

    logger.info(
        "manufacturing_incident_analysis_started"
    )

    question = missing_core_question(request.incident)
    if question:
        return IncidentAnalysisResponse(
            summary="More incident detail is needed.",
            severity="LOW",
            estimated_loss_usd=0,
            likely_causes=[],
            immediate_actions=[],
            verification_checks=[],
            escalation_required=False,
            confidence=0,
            needs_clarification=True,
            clarification_question=question,
        )

    try:

        # -------------------------------------------------
        # RAG RETRIEVAL
        # -------------------------------------------------

        knowledge_results = retrieve_knowledge(
            request.incident
        )

        logger.info(
            f"rag_documents_retrieved "
            f"count={len(knowledge_results)}"
        )

        knowledge_context = "\n\n".join(
            [
                (
                    f"SOURCE: {item['source']}\n"
                    f"RETRIEVAL SCORE: "
                    f"{item['score']}\n"
                    f"{item['content']}"
                )
                for item in knowledge_results
            ]
        )

        if not knowledge_context:

            knowledge_context = (
                "No relevant manufacturing "
                "knowledge document was found."
            )

        # -------------------------------------------------
        # WEB SEARCH (deterministic, always runs)
        # Both searches run concurrently in threads instead
        # of one after another, since search_web() is a
        # blocking network call.
        # -------------------------------------------------

        equipment_term = equipment_search_term(request.incident)
        incident_terms = " ".join(request.incident.split())

        (
            web_results,
            supplier_results,
            cost_results,
        ) = await asyncio.gather(
            asyncio.to_thread(
                search_web,
                f"{equipment_term} {incident_terms} "
                "industrial troubleshooting",
            ),
            asyncio.to_thread(
                search_web,
                f"{equipment_term} manufacturer "
                "specifications service support",
            ),
            asyncio.to_thread(
                search_web,
                f"{equipment_term} spare parts price cost buy",
            ),
        )

        logger.info(
            f"web_results_retrieved "
            f"count={len(web_results)}"
        )

        logger.info(
            f"supplier_results_retrieved "
            f"count={len(supplier_results)}"
        )

        logger.info(
            f"cost_results_retrieved "
            f"count={len(cost_results)}"
        )

        web_context = "\n\n".join(
            [
                (
                    f"TITLE: {item['title']}\n"
                    f"URL: {item['url']}\n"
                    f"{item['snippet']}"
                )
                for item in web_results
            ]
        )

        if not web_context:

            web_context = (
                "No relevant web search "
                "results were found."
            )

        supplier_context = "\n\n".join(
            [
                (
                    f"TITLE: {item['title']}\n"
                    f"URL: {item['url']}\n"
                    f"{item['snippet']}"
                )
                for item in supplier_results
            ]
        )

        if not supplier_context:

            supplier_context = (
                "No relevant equipment "
                "specification/supplier web "
                "results were found."
            )

        cost_context = "\n\n".join(
            [
                (
                    f"TITLE: {item['title']}\n"
                    f"URL: {item['url']}\n"
                    f"{item['snippet']}"
                )
                for item in cost_results
            ]
        )

        if not cost_context:

            cost_context = (
                "No relevant spare-parts pricing "
                "web results were found."
            )

        # -------------------------------------------------
        # AGENT PROMPT
        # -------------------------------------------------

        prompt = f"""
Analyze the following manufacturing incident.

The incident text may contain several successive user
messages about the SAME incident. Combine their facts.
Later messages may identify the equipment more precisely.
An equipment type and a symptom are enough to give a
provisional analysis. A model number, alarm code, and
measurements are useful verification details, not
prerequisites. Ask a question only if the equipment type
or symptom remains unknown after combining all messages.

====================================================
USER-PROVIDED INCIDENT
====================================================

INCIDENT:
{request.incident}

DOWNTIME:
{request.downtime_minutes} minutes

PRODUCTION LOSS RATE:
{request.loss_rate_usd_per_hour} USD/hour


====================================================
RETRIEVED MANUFACTURING KNOWLEDGE
====================================================

{knowledge_context}


====================================================
RETRIEVED WEB SEARCH RESULTS
====================================================

{web_context}


====================================================
RETRIEVED EQUIPMENT SPECIFICATION / SUPPLIER
WEB SEARCH RESULTS
====================================================

{supplier_context}


====================================================
RETRIEVED SPARE-PARTS PRICING WEB SEARCH RESULTS
====================================================

{cost_context}


====================================================
RULES FOR USING RETRIEVED KNOWLEDGE
====================================================

1. Treat retrieved knowledge and web search results as
   reference evidence.

2. Neither retrieved knowledge nor web search results
   prove that a listed component or failure mode
   actually failed at this site.

3. Possible causes from retrieved knowledge or web search
   results must remain hypotheses until inspection or
   measurement verifies them.

4. Use relevant safety instructions from retrieved
   manufacturing knowledge or web search results.

5. Do not invent information that is absent from the
   incident, retrieved knowledge, and web search results.

6. Clearly distinguish between:
   - user-provided facts
   - deterministic tool results
   - retrieved local knowledge
   - retrieved web search results
   - hypotheses requiring verification

7. If neither relevant local knowledge nor relevant web
   results were found, keep equipment-specific analysis
   generic.

8. Use the available deterministic tools where required.

9. Fill in equipment_specifications,
   supplier_availability, and spare_parts_cost_estimate
   using only the retrieved local knowledge and the
   retrieved web search results above:

   - First confirm the equipment type is explicitly named
     or clearly described in the INCIDENT text itself.
     Do not assume an equipment type merely because it is
     what the retrieved knowledge or web results happen to
     be about.

   - If the equipment type is not clearly identifiable
     from the incident text, set
     equipment_specifications to state that the equipment
     type is not identified and ask the user to confirm or
     specify it, and set supplier_availability to an empty
     list. Do not fall back to whatever equipment the
     retrieved documents happen to describe.

   - equipment_specifications: a short factual summary of
     the equipment's typical specifications (type, model
     family, capacity range, etc.) if the equipment is
     identifiable and the evidence supports it. If the
     evidence is insufficient, state that plainly instead
     of guessing exact numbers.

   - supplier_availability: a list of manufacturers or
     suppliers of this equipment type and, where the
     evidence indicates it, the world regions they serve
     or are known to operate in. Only include a supplier
     if it is actually named in the retrieved knowledge or
     web results above. If none are found, return an empty
     list rather than inventing supplier names.

   - spare_parts_cost_estimate: a short summary of any
     spare-parts or equipment price/cost figures found in
     the retrieved spare-parts pricing web search results
     above, including the currency and source context
     exactly as reported (e.g. "Listings found: seal kits
     roughly $150-400 USD per aftermarket listings; verify
     against your own supplier quote"). Never invent a
     number. If no pricing web result is relevant, state
     plainly that no cost information was found rather
     than estimating one.

10. If the incident is too ambiguous or incomplete to
    analyze confidently, set needs_clarification to true
    and ask exactly one focused clarification_question.
    Do not ask for an exact model, alarm, or measurement
    when equipment and symptom are already provided.
    Otherwise set needs_clarification to false and leave
    clarification_question null.

11. Return a structured manufacturing incident
    action plan.
"""

        # -------------------------------------------------
        # RUN AGENT
        # -------------------------------------------------

        result = await run_agent_with_retry(
            prompt=prompt,
            max_attempts=(
                settings.MAX_AGENT_ATTEMPTS
            ),
        )

        # -------------------------------------------------
        # DETERMINISTIC SAFETY GUARDRAILS
        # -------------------------------------------------

        validated_plan = (
            apply_safety_guardrails(
                request=request,
                plan=result.final_output,
            )
        )

        # The model sometimes treats optional model numbers
        # or alarm codes as mandatory. Core incident facts
        # may arrive in separate chat turns.
        if has_core_incident_details(request.incident):
            validated_plan.needs_clarification = False
            validated_plan.clarification_question = None

        # -------------------------------------------------
        # DETERMINISTIC KNOWLEDGE SOURCE ATTACHMENT
        # -------------------------------------------------

        validated_plan.knowledge_sources = sorted(
            {
                item["source"]
                for item in knowledge_results
                if item.get("source")
            }
        )

        logger.info(
            f"knowledge_sources_attached "
            f"count="
            f"{len(validated_plan.knowledge_sources)}"
        )

        logger.info(
            f"web_sources_attached "
            f"count={len(web_results)}"
        )

        # -------------------------------------------------
        # DATABASE PERSISTENCE
        # -------------------------------------------------

        incident_id = save_incident(
            request=request,
            plan=validated_plan,
        )

        logger.info(
            f"incident_saved "
            f"incident_id={incident_id}"
        )

        # -------------------------------------------------
        # CHART / VISUAL GENERATION
        # -------------------------------------------------

        severity_dashboard_image = (
            build_severity_dashboard(
                severity=validated_plan.severity,
                confidence=validated_plan.confidence,
                escalation_required=(
                    validated_plan.escalation_required
                ),
            )
        )

        failure_trend_image = (
            build_failure_trend_chart(
                validated_plan.knowledge_sources
            )
        )

        failure_count_image = (
            build_failure_count_chart(
                validated_plan.knowledge_sources
            )
        )

        pareto_analysis = (
            build_pareto_analysis(
                validated_plan.knowledge_sources
            )
        )

        response = IncidentAnalysisResponse(
            **validated_plan.model_dump(),
            severity_dashboard_image=(
                severity_dashboard_image
            ),
            failure_trend_image=(
                failure_trend_image
            ),
            failure_count_image=(
                failure_count_image
            ),
            web_sources=[
                {
                    "title": item["title"],
                    "url": item["url"],
                }
                for item in web_results + supplier_results
                if item.get("url")
            ],
            pareto_analysis=pareto_analysis,
        )

        logger.info(
            "manufacturing_incident_analysis_completed"
        )

        return response

    except asyncio.TimeoutError:

        logger.error(
            "manufacturing_incident_analysis_timeout"
        )

        raise HTTPException(
            status_code=504,
            detail=(
                "Manufacturing agent timed out. "
                "Please try again."
            ),
        )

    except AgentProviderError:

        logger.exception(
            "manufacturing_incident_provider_failure"
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "AI provider temporarily failed. "
                "Please try again."
            ),
        )

    except AgentExecutionError:

        logger.exception(
            "manufacturing_incident_agent_failure"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Manufacturing agent execution failed."
            ),
        )

    except HTTPException:

        raise

    except Exception:

        logger.exception(
            "manufacturing_incident_unexpected_failure"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected internal server error."
            ),
        )


# =========================================================
# INCIDENT HISTORY
# =========================================================

@app.get("/incidents")
async def list_incidents(
    limit: int = 50,
    authorized: bool = Depends(
        verify_api_key
    ),
):

    if limit < 1:

        raise HTTPException(
            status_code=400,
            detail=(
                "Limit must be at least 1."
            ),
        )

    if limit > 100:
        limit = 100

    incidents = get_incidents(
        limit=limit
    )

    return {
        "count": len(incidents),
        "incidents": incidents,
    }


# =========================================================
# GET INCIDENT BY ID
# =========================================================

@app.get("/incidents/{incident_id}")
async def get_incident(
    incident_id: int,
    authorized: bool = Depends(
        verify_api_key
    ),
):

    if incident_id < 1:

        raise HTTPException(
            status_code=400,
            detail=(
                "Incident ID must be "
                "greater than 0."
            ),
        )

    incident = get_incident_by_id(
        incident_id=incident_id
    )

    if incident is None:

        raise HTTPException(
            status_code=404,
            detail="Incident not found.",
        )

    return incident
