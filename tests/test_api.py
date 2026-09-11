import asyncio
import os

from types import SimpleNamespace
from unittest.mock import (
    AsyncMock,
    patch,
)

from dotenv import load_dotenv
from fastapi.testclient import TestClient

from app.main import (
    AgentExecutionError,
    AgentProviderError,
    app,
)

from app.models import IncidentActionPlan


# =========================================================
# TEST SETUP
# =========================================================

load_dotenv()

API_KEY = os.getenv(
    "APP_API_KEY"
)

client = TestClient(
    app
)


# =========================================================
# HEALTH TEST
# =========================================================

def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["status"]
        == "healthy"
    )

    assert (
        data["service"]
        == "manufacturing-operations-agent"
    )


# =========================================================
# AUTHENTICATION TESTS
# =========================================================

def test_analyze_without_api_key_rejected():

    response = client.post(
        "/analyze",
        json={
            "incident": (
                "PET cutter stopped with "
                "abnormal vibration."
            ),
            "downtime_minutes": 120,
            "loss_rate_usd_per_hour": 500,
        },
    )

    assert response.status_code == 401


def test_analyze_with_wrong_api_key_rejected():

    response = client.post(
        "/analyze",
        headers={
            "X-API-Key": "wrong-key"
        },
        json={
            "incident": (
                "PET cutter stopped with "
                "abnormal vibration."
            ),
            "downtime_minutes": 120,
            "loss_rate_usd_per_hour": 500,
        },
    )

    assert response.status_code == 403


# =========================================================
# REQUEST VALIDATION TESTS
# =========================================================

def test_invalid_request():

    response = client.post(
        "/analyze",
        headers={
            "X-API-Key": API_KEY
        },
        json={},
    )

    assert response.status_code == 422


def test_negative_downtime_rejected():

    response = client.post(
        "/analyze",
        headers={
            "X-API-Key": API_KEY
        },
        json={
            "incident": (
                "PET cutter stopped with "
                "abnormal vibration."
            ),
            "downtime_minutes": -10,
            "loss_rate_usd_per_hour": 500,
        },
    )

    assert response.status_code == 422


def test_negative_loss_rate_rejected():

    response = client.post(
        "/analyze",
        headers={
            "X-API-Key": API_KEY
        },
        json={
            "incident": (
                "PET cutter stopped with "
                "abnormal vibration."
            ),
            "downtime_minutes": 120,
            "loss_rate_usd_per_hour": -500,
        },
    )

    assert response.status_code == 422


def test_short_incident_rejected():

    response = client.post(
        "/analyze",
        headers={
            "X-API-Key": API_KEY
        },
        json={
            "incident": "short",
            "downtime_minutes": 120,
            "loss_rate_usd_per_hour": 500,
        },
    )

    assert response.status_code == 422


def test_meaningless_one_word_incident_rejected():

    response = client.post(
        "/analyze",
        headers={
            "X-API-Key": API_KEY
        },
        json={
            "incident": "stringstri",
            "downtime_minutes": 120,
            "loss_rate_usd_per_hour": 500,
        },
    )

    assert response.status_code == 422


# =========================================================
# PROVIDER FAILURE TEST
# =========================================================

def test_provider_failure_returns_502():

    with patch(
        "app.main.run_agent_with_retry",
        new_callable=AsyncMock,
    ) as mock_run:

        mock_run.side_effect = (
            AgentProviderError(
                "OpenRouter failure"
            )
        )

        response = client.post(
            "/analyze",
            headers={
                "X-API-Key": API_KEY
            },
            json={
                "incident": (
                    "PET cutter tripped repeatedly "
                    "with abnormal vibration."
                ),
                "downtime_minutes": 120,
                "loss_rate_usd_per_hour": 500,
            },
        )

    assert response.status_code == 502

    assert (
        response.json()["detail"]
        == (
            "AI provider temporarily failed. "
            "Please try again."
        )
    )


# =========================================================
# TIMEOUT TEST
# =========================================================

def test_timeout_returns_504():

    with patch(
        "app.main.run_agent_with_retry",
        new_callable=AsyncMock,
    ) as mock_run:

        mock_run.side_effect = (
            asyncio.TimeoutError()
        )

        response = client.post(
            "/analyze",
            headers={
                "X-API-Key": API_KEY
            },
            json={
                "incident": (
                    "PET cutter tripped repeatedly "
                    "with abnormal vibration."
                ),
                "downtime_minutes": 120,
                "loss_rate_usd_per_hour": 500,
            },
        )

    assert response.status_code == 504

    assert (
        response.json()["detail"]
        == (
            "Manufacturing agent timed out. "
            "Please try again."
        )
    )


# =========================================================
# AGENT EXECUTION FAILURE TEST
# =========================================================

def test_agent_failure_returns_500():

    with patch(
        "app.main.run_agent_with_retry",
        new_callable=AsyncMock,
    ) as mock_run:

        mock_run.side_effect = (
            AgentExecutionError(
                "Unexpected agent error"
            )
        )

        response = client.post(
            "/analyze",
            headers={
                "X-API-Key": API_KEY
            },
            json={
                "incident": (
                    "PET cutter tripped repeatedly "
                    "with abnormal vibration."
                ),
                "downtime_minutes": 120,
                "loss_rate_usd_per_hour": 500,
            },
        )

    assert response.status_code == 500

    assert (
        response.json()["detail"]
        == (
            "Manufacturing agent "
            "execution failed."
        )
    )


# =========================================================
# SUCCESSFUL MOCKED AGENT TEST
# =========================================================

def test_successful_analyze_with_mocked_agent():

    fake_plan = IncidentActionPlan(
        summary=(
            "PET Line 6 cutter tripped repeatedly "
            "with abnormal vibration."
        ),
        severity="MEDIUM",
        estimated_loss_usd=1000,
        likely_causes=[
            "Possible mechanical issue"
        ],
        immediate_actions=[
            "Inspect the cutter"
        ],
        verification_checks=[
            "Check vibration"
        ],
        escalation_required=False,
        confidence=0.80,
    )

    fake_result = SimpleNamespace(
        final_output=fake_plan
    )

    with patch(
        "app.main.run_agent_with_retry",
        new_callable=AsyncMock,
    ) as mock_run:

        mock_run.return_value = (
            fake_result
        )

        response = client.post(
            "/analyze",
            headers={
                "X-API-Key": API_KEY
            },
            json={
                "incident": (
                    "PET Line 6 cutter "
                    "tripped repeatedly with "
                    "abnormal vibration."
                ),
                "downtime_minutes": 120,
                "loss_rate_usd_per_hour": 500,
            },
        )

        assert (
            response.status_code
            == 200
        )

        data = response.json()

        assert (
            data["estimated_loss_usd"]
            == 1000
        )

        # Guardrail should upgrade
        # MEDIUM to HIGH.
        assert (
            data["severity"]
            == "HIGH"
        )

        assert (
            data["escalation_required"]
            is True
        )

        required_action = (
            "Keep the affected equipment stopped "
            "until maintenance or engineering has "
            "completed a safe inspection."
        )

        assert (
            required_action
            in data["immediate_actions"]
        )

        mock_run.assert_awaited_once()


# =========================================================
# RAG INTEGRATION TEST
# =========================================================

def test_rag_context_is_added_to_agent_prompt():

    fake_plan = IncidentActionPlan(
        summary=(
            "Cutter incident requiring "
            "inspection."
        ),
        severity="HIGH",
        estimated_loss_usd=1000,
        likely_causes=[
            (
                "Mechanical issue "
                "requiring inspection"
            )
        ],
        immediate_actions=[
            "Keep equipment stopped"
        ],
        verification_checks=[
            "Inspect cutter"
        ],
        escalation_required=True,
        confidence=0.8,
    )

    fake_result = SimpleNamespace(
        final_output=fake_plan
    )

    with patch(
        "app.main.run_agent_with_retry",
        new_callable=AsyncMock,
    ) as mock_run:

        mock_run.return_value = (
            fake_result
        )

        response = client.post(
            "/analyze",
            headers={
                "X-API-Key": API_KEY
            },
            json={
                "incident": (
                    "PET cutter tripped repeatedly "
                    "with abnormal vibration."
                ),
                "downtime_minutes": 120,
                "loss_rate_usd_per_hour": 500,
            },
        )

        assert (
            response.status_code
            == 200
        )

        call_args = (
            mock_run.await_args.kwargs
        )

        prompt = (
            call_args["prompt"]
        )

        assert (
            "cutter_troubleshooting.txt"
            in prompt
        )

        assert (
            "PET CUTTER "
            "TROUBLESHOOTING KNOWLEDGE"
            in prompt
        )

        assert (
            "RETRIEVED MANUFACTURING KNOWLEDGE"
            in prompt
        )