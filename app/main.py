import asyncio
import logging
import time
import uuid

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
)

from agents import Runner

from app.agent import operations_agent
from app.config import settings
from app.database import (
    get_incident_by_id,
    get_incidents,
    initialize_database,
    save_incident,
)
from app.guardrails import apply_safety_guardrails
from app.logging_config import setup_logging
from app.models import (
    IncidentActionPlan,
    IncidentRequest,
)
from app.rag import retrieve_knowledge
from app.security import verify_api_key


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
    }


# =========================================================
# INCIDENT ANALYSIS
# =========================================================

@app.post(
    "/analyze",
    response_model=IncidentActionPlan,
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
        # AGENT PROMPT
        # -------------------------------------------------

        prompt = f"""
Analyze the following manufacturing incident.

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
RULES FOR USING RETRIEVED KNOWLEDGE
====================================================

1. Treat retrieved knowledge as reference evidence.

2. Retrieved knowledge does not prove that a listed
   component or failure mode actually failed.

3. Possible causes from retrieved knowledge must remain
   hypotheses until inspection or measurement verifies them.

4. Use relevant safety instructions from retrieved
   manufacturing knowledge.

5. Do not invent information that is absent from both
   the incident and retrieved knowledge.

6. Clearly distinguish between:
   - user-provided facts
   - deterministic tool results
   - retrieved knowledge
   - hypotheses requiring verification

7. If no relevant knowledge was retrieved, keep
   equipment-specific analysis generic.

8. Use the available deterministic tools where required.

9. Return a structured manufacturing incident
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

        logger.info(
            "manufacturing_incident_analysis_completed"
        )

        return validated_plan

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