import asyncio
import os

from types import SimpleNamespace
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

from dotenv import load_dotenv
from fastapi.testclient import TestClient

from app.main import (
    AgentExecutionError,
    AgentProviderError,
    app,
    equipment_search_term,
    has_core_incident_details,
    run_agent_with_retry,
)

from app.models import (
    ChatDecision,
    IncidentActionPlan,
    IncidentAnalysisResponse,
)


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


def test_ollama_invalid_json_uses_native_schema_fallback():

    plan = IncidentActionPlan(
        summary="Polymer filter condition is unclear.",
        severity="MEDIUM",
        estimated_loss_usd=0,
        likely_causes=["Condition requires inspection."],
        immediate_actions=["Inspect safely."],
        verification_checks=["Check differential pressure."],
        escalation_required=False,
        confidence=0.4,
        needs_clarification=True,
        clarification_question="What symptom is present?",
    )
    response = MagicMock()
    response.json.return_value = {
        "message": {"content": plan.model_dump_json()},
        "done_reason": "stop",
        "eval_count": 100,
    }

    with (
        patch("app.main.settings.AI_PROVIDER", "ollama"),
        patch("app.main.Runner.run", new_callable=AsyncMock) as run,
        patch("app.main.httpx.AsyncClient") as client_type,
    ):
        run.side_effect = Exception("Invalid JSON when parsing model output")
        client_type.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=response
        )

        result = asyncio.run(
            run_agent_with_retry("Analyze polymer filter", max_attempts=1)
        )

    assert result.final_output.needs_clarification is True
    assert result.final_output.summary == plan.summary


def test_followup_details_complete_same_incident():
    incident = "not running\npump\nMaag pump"
    assert has_core_incident_details(incident)
    assert equipment_search_term(incident) == "Maag pump"

    plan = IncidentActionPlan(
        summary="MAAG pump is not running.",
        severity="MEDIUM",
        estimated_loss_usd=0,
        likely_causes=["Electrical or mechanical issue."],
        immediate_actions=["Keep the pump stopped."],
        verification_checks=["Check power and alarms."],
        escalation_required=False,
        confidence=0.5,
        needs_clarification=True,
        clarification_question="What is the pump model?",
    )
    web_result = [{
        "title": "MAAG pump service",
        "url": "https://example.com/maag-pump",
        "snippet": "Manufacturer service guidance",
    }]

    with (
        patch("app.main.retrieve_knowledge", return_value=[]),
        patch("app.main.search_web", return_value=web_result) as search,
        patch("app.main.run_agent_with_retry", new_callable=AsyncMock)
        as run,
    ):
        run.return_value = SimpleNamespace(final_output=plan)
        response = client.post(
            "/analyze",
            headers={"X-API-Key": API_KEY},
            json={"incident": incident},
        )

    assert response.status_code == 200
    assert response.json()["needs_clarification"] is False
    assert response.json()["clarification_question"] is None
    assert response.json()["web_sources"]
    assert "Maag pump" in search.call_args_list[0].args[0]


def test_equipment_only_asks_for_symptom_without_analysis():
    with (
        patch("app.main.retrieve_knowledge") as retrieve,
        patch("app.main.search_web") as search,
        patch("app.main.run_agent_with_retry") as run,
    ):
        response = client.post(
            "/analyze",
            headers={"X-API-Key": API_KEY},
            json={"incident": "polymer pump"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["needs_clarification"] is True
    assert "symptom" in data["clarification_question"].lower()
    assert data["likely_causes"] == []
    retrieve.assert_not_called()
    search.assert_not_called()
    run.assert_not_called()


def test_pump_image_request_uses_chat_route():
    model_response = MagicMock()
    model_response.json.return_value = {
        "message": {
            "content": ChatDecision(
                action="pump_diagram",
                reply="Here is a pump diagram.",
            ).model_dump_json()
        }
    }
    with (
        patch("app.main.settings.AI_PROVIDER", "ollama"),
        patch("app.main.httpx.AsyncClient") as client_type,
    ):
        client_type.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=model_response
        )
        response = client.post(
            "/chat",
            headers={"X-API-Key": API_KEY},
            json={"message": "develop one image for the pump"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "pump" in data["reply"].lower()
    assert data["image_svg"].startswith("<svg")
    assert "<svg" not in data["reply"]


def test_chat_respects_a_conversational_no():
    model_response = MagicMock()
    model_response.json.return_value = {
        "message": {
            "content": ChatDecision(
                action="reply",
                reply="No problem. What else can I help with?",
            ).model_dump_json()
        }
    }
    with (
        patch("app.main.settings.AI_PROVIDER", "ollama"),
        patch("app.main.httpx.AsyncClient") as client_type,
    ):
        client_type.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=model_response
        )
        response = client.post(
            "/chat",
            headers={"X-API-Key": API_KEY},
            json={"message": "no"},
        )

    assert response.status_code == 200
    assert "No problem" in response.json()["reply"]
    assert response.json()["analysis"] is None


def test_chat_uses_analysis_after_equipment_followup():
    model_response = MagicMock()
    model_response.json.return_value = {
        "message": {
            "content": ChatDecision(
                action="reply",
                reply="Can you provide an alarm code?",
            ).model_dump_json()
        }
    }
    plan = IncidentAnalysisResponse(
        summary="Pump is not running.",
        severity="MEDIUM",
        estimated_loss_usd=0,
        likely_causes=["Electrical issue requiring inspection."],
        immediate_actions=["Keep pump stopped."],
        verification_checks=["Check supply voltage."],
        escalation_required=False,
        confidence=0.5,
    )
    with (
        patch("app.main.settings.AI_PROVIDER", "ollama"),
        patch("app.main.httpx.AsyncClient") as client_type,
        patch("app.main.analyze_incident", new_callable=AsyncMock)
        as analyze,
    ):
        client_type.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=model_response
        )
        analyze.return_value = plan
        response = client.post(
            "/chat",
            headers={"X-API-Key": API_KEY},
            json={
                "message": "pump",
                "history": [{"role": "user", "content": "not running"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["analysis"]["summary"] == plan.summary
    assert analyze.await_args.args[0].incident == "not running\npump"


def test_chat_uses_knowledge_workbook_for_pareto_followup():
    model_response = MagicMock()
    model_response.json.return_value = {
        "message": {"content": ChatDecision(
            action="reply", reply="I don't have access to past data."
        ).model_dump_json()}
    }
    with (
        patch("app.main.settings.AI_PROVIDER", "ollama"),
        patch("app.main.httpx.AsyncClient") as client_type,
    ):
        client_type.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=model_response
        )
        response = client.post(
            "/chat",
            headers={"X-API-Key": API_KEY},
            json={
                "message": "Based on the past data",
                "history": [{"role": "user", "content": "Can you develop a Pareto diagram for the cutter trip?"}],
            },
        )

    assert response.status_code == 200
    pareto = response.json()["pareto"]
    assert pareto["source"] == "2026 cutter benchmarking IVL.xlsx"
    assert pareto["total_reports"] == 13
    assert pareto["image"].startswith("data:image/png;base64,")
    assert "not trip event counts" in response.json()["reply"]


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
