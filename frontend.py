import os

import requests
import streamlit as st
from dotenv import load_dotenv


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

API_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000",
)

APP_API_KEY = os.getenv(
    "APP_API_KEY",
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Manufacturing Operations AI Agent",
    page_icon="🏭",
    layout="wide",
)


# =========================================================
# VALIDATE CONFIGURATION
# =========================================================

if not APP_API_KEY:
    st.error(
        "APP_API_KEY is not configured."
    )
    st.stop()


# =========================================================
# HEADER
# =========================================================

st.title(
    "🏭 Manufacturing Operations AI Agent"
)

st.caption(
    (
        "Analyze manufacturing incidents using "
        "Agentic AI, deterministic tools, RAG, "
        "and safety guardrails."
    )
)


# =========================================================
# SIDEBAR - SYSTEM STATUS
# =========================================================

with st.sidebar:

    st.header(
        "System Status"
    )

    backend_online = False

    try:

        health_response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        if (
            health_response.status_code
            == 200
        ):

            backend_online = True

            st.success(
                "Backend Online"
            )

        else:

            st.warning(
                (
                    "Backend responded with "
                    f"HTTP {health_response.status_code}"
                )
            )

    except requests.exceptions.RequestException:

        st.error(
            "Backend Offline"
        )

    st.caption(
        f"Backend: {API_URL}"
    )

    st.divider()

    st.subheader(
        "Architecture"
    )

    st.markdown(
        """
        **Frontend**
        - Streamlit

        **Backend**
        - FastAPI

        **Agent**
        - OpenAI Agents SDK

        **LLM**
        - OpenRouter

        **RAG**
        - Sentence Transformers
        - Semantic retrieval
        - Persistent vector index

        **Safety**
        - Deterministic guardrails

        **Persistence**
        - SQLite
        """
    )


# =========================================================
# INCIDENT INPUT
# =========================================================

st.subheader(
    "Incident Details"
)

incident = st.text_area(
    "Incident description",
    value=(
        "PET Line 6 cutter tripped four times today "
        "with abnormal vibration."
    ),
    height=140,
    help=(
        "Describe the manufacturing incident, "
        "equipment condition, alarms, symptoms, "
        "or operational problem."
    ),
)


# =========================================================
# NUMERIC INPUTS
# =========================================================

col1, col2 = st.columns(2)

with col1:

    downtime_minutes = st.number_input(
        "Downtime (minutes)",
        min_value=0.0,
        value=120.0,
        step=10.0,
        help=(
            "Total downtime caused by the incident."
        ),
    )


with col2:

    loss_rate = st.number_input(
        "Loss rate (USD/hour)",
        min_value=0.0,
        value=500.0,
        step=50.0,
        help=(
            "Estimated financial loss per hour "
            "of downtime."
        ),
    )


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze_clicked = st.button(
    "Analyze Incident",
    type="primary",
    use_container_width=True,
)


# =========================================================
# ANALYSIS REQUEST
# =========================================================

if analyze_clicked:

    # -----------------------------------------------------
    # FRONTEND VALIDATION
    # -----------------------------------------------------

    cleaned_incident = (
        incident.strip()
    )

    if not cleaned_incident:

        st.error(
            "Please enter an incident description."
        )

        st.stop()

    if len(
        cleaned_incident.split()
    ) < 3:

        st.error(
            (
                "Please provide a more meaningful "
                "incident description."
            )
        )

        st.stop()

    if not backend_online:

        st.error(
            (
                "The backend is currently offline. "
                "Start the FastAPI service before "
                "analyzing an incident."
            )
        )

        st.stop()


    # -----------------------------------------------------
    # API PAYLOAD
    # -----------------------------------------------------

    payload = {
        "incident": (
            cleaned_incident
        ),
        "downtime_minutes": (
            downtime_minutes
        ),
        "loss_rate_usd_per_hour": (
            loss_rate
        ),
    }


    # -----------------------------------------------------
    # API HEADERS
    # -----------------------------------------------------

    headers = {
        "X-API-Key": (
            APP_API_KEY
        )
    }


    # -----------------------------------------------------
    # CALL FASTAPI
    # -----------------------------------------------------

    try:

        with st.spinner(
            "Analyzing incident..."
        ):

            response = requests.post(
                f"{API_URL}/analyze",
                json=payload,
                headers=headers,
                timeout=120,
            )


    except requests.exceptions.ConnectionError:

        st.error(
            (
                "Cannot connect to the backend. "
                f"Backend URL: {API_URL}"
            )
        )

        st.stop()


    except requests.exceptions.Timeout:

        st.error(
            (
                "The analysis request timed out. "
                "Please try again."
            )
        )

        st.stop()


    except requests.exceptions.RequestException as error:

        st.error(
            f"Backend request failed: {error}"
        )

        st.stop()


    # =====================================================
    # HANDLE API ERROR
    # =====================================================

    if response.status_code != 200:

        try:

            error_data = (
                response.json()
            )

            detail = (
                error_data.get(
                    "detail",
                    error_data,
                )
            )

        except Exception:

            detail = (
                response.text
            )

        st.error(
            (
                f"API Error "
                f"({response.status_code}): "
                f"{detail}"
            )
        )

        st.stop()


    # =====================================================
    # READ RESPONSE
    # =====================================================

    try:

        result = (
            response.json()
        )

    except ValueError:

        st.error(
            (
                "The backend returned an "
                "invalid JSON response."
            )
        )

        st.stop()


    # =====================================================
    # INCIDENT ASSESSMENT
    # =====================================================

    st.divider()

    st.subheader(
        "Incident Assessment"
    )


    # =====================================================
    # KPI CARDS
    # =====================================================

    col1, col2, col3, col4 = (
        st.columns(4)
    )


    # -----------------------------------------------------
    # SEVERITY
    # -----------------------------------------------------

    severity = result.get(
        "severity",
        "N/A",
    )

    with col1:

        st.metric(
            "Severity",
            severity,
        )


    # -----------------------------------------------------
    # ESTIMATED LOSS
    # -----------------------------------------------------

    estimated_loss = result.get(
        "estimated_loss_usd"
    )

    if estimated_loss is None:

        loss_display = "N/A"

    else:

        loss_display = (
            f"${estimated_loss:,.2f}"
        )

    with col2:

        st.metric(
            "Estimated Loss",
            loss_display,
        )


    # -----------------------------------------------------
    # ESCALATION
    # -----------------------------------------------------

    escalation = result.get(
        "escalation_required",
        False,
    )

    with col3:

        st.metric(
            "Escalation",
            (
                "YES"
                if escalation
                else "NO"
            ),
        )


    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    confidence = result.get(
        "confidence"
    )

    if confidence is None:

        confidence_display = (
            "N/A"
        )

    else:

        confidence_display = (
            f"{confidence * 100:.0f}%"
        )

    with col4:

        st.metric(
            "Confidence",
            confidence_display,
        )


    # =====================================================
    # SAFETY / ESCALATION MESSAGE
    # =====================================================

    if (
        severity
        == "CRITICAL"
    ):

        st.error(
            (
                "CRITICAL INCIDENT — "
                "Immediate escalation is required."
            )
        )

    elif (
        severity
        == "HIGH"
    ):

        st.warning(
            (
                "HIGH-SEVERITY INCIDENT — "
                "Maintenance or engineering "
                "attention is recommended."
            )
        )

    elif escalation:

        st.warning(
            "Escalation is required."
        )


    # =====================================================
    # SUMMARY
    # =====================================================

    st.subheader(
        "Summary"
    )

    st.write(
        result.get(
            "summary",
            "No summary returned.",
        )
    )


    # =====================================================
    # LIKELY CAUSES
    # =====================================================

    st.subheader(
        "Likely Causes"
    )

    likely_causes = result.get(
        "likely_causes",
        [],
    )

    if likely_causes:

        for item in likely_causes:

            st.markdown(
                f"- {item}"
            )

    else:

        st.info(
            "No likely causes returned."
        )


    # =====================================================
    # IMMEDIATE ACTIONS
    # =====================================================

    st.subheader(
        "Immediate Actions"
    )

    immediate_actions = (
        result.get(
            "immediate_actions",
            [],
        )
    )

    if immediate_actions:

        for index, item in enumerate(
            immediate_actions,
            start=1,
        ):

            st.markdown(
                f"{index}. {item}"
            )

    else:

        st.info(
            "No immediate actions returned."
        )


    # =====================================================
    # VERIFICATION CHECKS
    # =====================================================

    st.subheader(
        "Verification Checks"
    )

    verification_checks = (
        result.get(
            "verification_checks",
            [],
        )
    )

    if verification_checks:

        for item in verification_checks:

            st.markdown(
                f"- {item}"
            )

    else:

        st.info(
            "No verification checks returned."
        )


    # =====================================================
    # KNOWLEDGE SOURCES
    # =====================================================

    st.subheader(
        "Knowledge Sources"
    )

    knowledge_sources = (
        result.get(
            "knowledge_sources",
            [],
        )
    )

    if knowledge_sources:

        for source in knowledge_sources:

            st.code(
                source
            )

    else:

        st.info(
            (
                "No knowledge source "
                "was retrieved."
            )
        )


    # =====================================================
    # RAW RESPONSE
    # =====================================================

    with st.expander(
        "View raw API response"
    ):

        st.json(
            result
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    (
        "Manufacturing Operations AI Agent • "
        "FastAPI + Agentic AI + Semantic RAG + "
        "Deterministic Tools + Safety Guardrails"
    )
)