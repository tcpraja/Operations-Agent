
from app.models import (
    IncidentRequest,
    IncidentActionPlan,
)

from app.guardrails import (
    apply_safety_guardrails,
)


def test_trip_and_vibration_force_high_severity():

    request = IncidentRequest(
        incident=(
            "PET Line 6 cutter tripped repeatedly "
            "with abnormal vibration."
        ),
        downtime_minutes=120,
        loss_rate_usd_per_hour=500,
    )

    plan = IncidentActionPlan(
        summary="Test incident",
        severity="MEDIUM",
        estimated_loss_usd=1000,
        likely_causes=[
            "Possible mechanical issue"
        ],
        immediate_actions=[
            "Inspect the equipment"
        ],
        verification_checks=[
            "Check vibration"
        ],
        escalation_required=False,
        confidence=0.70,
    )

    result = apply_safety_guardrails(
        request=request,
        plan=plan,
    )

    assert result.severity == "HIGH"
    assert result.escalation_required is True

    assert (
        "Keep the affected equipment stopped "
        "until maintenance or engineering has "
        "completed a safe inspection."
        in result.immediate_actions
    )


def test_critical_severity_forces_escalation():

    request = IncidentRequest(
        incident=(
            "Major manufacturing equipment incident "
            "requiring immediate attention."
        ),
        downtime_minutes=30,
        loss_rate_usd_per_hour=1000,
    )

    plan = IncidentActionPlan(
        summary="Critical incident",
        severity="CRITICAL",
        estimated_loss_usd=500,
        likely_causes=[
            "Cause under investigation"
        ],
        immediate_actions=[
            "Stop equipment"
        ],
        verification_checks=[
            "Inspect equipment"
        ],
        escalation_required=False,
        confidence=0.80,
    )

    result = apply_safety_guardrails(
        request=request,
        plan=plan,
    )

    assert result.severity == "CRITICAL"
    assert result.escalation_required is True


def test_normal_medium_incident_remains_medium():

    request = IncidentRequest(
        incident=(
            "Pump showed reduced discharge pressure "
            "during normal operation."
        ),
        downtime_minutes=20,
        loss_rate_usd_per_hour=200,
    )

    plan = IncidentActionPlan(
        summary="Pump issue",
        severity="MEDIUM",
        estimated_loss_usd=66.67,
        likely_causes=[
            "Possible suction issue"
        ],
        immediate_actions=[
            "Inspect pump"
        ],
        verification_checks=[
            "Check suction pressure"
        ],
        escalation_required=False,
        confidence=0.75,
    )

    result = apply_safety_guardrails(
        request=request,
        plan=plan,
    )

    assert result.severity == "MEDIUM"
    assert result.escalation_required is False