from app.models import (
    IncidentRequest,
    IncidentActionPlan,
)


def apply_safety_guardrails(
    request: IncidentRequest,
    plan: IncidentActionPlan,
) -> IncidentActionPlan:

    incident_text = request.incident.lower()

    # ---------------------------------------------------------
    # RULE 1:
    # Repeated trips + abnormal vibration must be escalated
    # ---------------------------------------------------------

    trip_signals = [
        "trip",
        "tripped",
        "multiple trips",
        "repeated trips",
        "tripping",
    ]

    vibration_signals = [
        "abnormal vibration",
        "high vibration",
        "excessive vibration",
        "vibration issue",
    ]

    has_trip = any(
        signal in incident_text
        for signal in trip_signals
    )

    has_vibration = any(
        signal in incident_text
        for signal in vibration_signals
    )

    if has_trip and has_vibration:

        plan.escalation_required = True

        if plan.severity in [
            "LOW",
            "MEDIUM",
        ]:
            plan.severity = "HIGH"

        required_action = (
            "Keep the affected equipment stopped "
            "until maintenance or engineering has "
            "completed a safe inspection."
        )

        if required_action not in plan.immediate_actions:
            plan.immediate_actions.insert(
                0,
                required_action,
            )

    # ---------------------------------------------------------
    # RULE 2:
    # If severity is CRITICAL, escalation must be required
    # ---------------------------------------------------------

    if plan.severity == "CRITICAL":
        plan.escalation_required = True

    # ---------------------------------------------------------
    # RULE 3:
    # Ensure confidence remains between 0 and 1
    # ---------------------------------------------------------

    if plan.confidence < 0:
        plan.confidence = 0

    if plan.confidence > 1:
        plan.confidence = 1

    # ---------------------------------------------------------
    # RULE 4:
    # Prevent negative loss values
    # ---------------------------------------------------------

    if (
        plan.estimated_loss_usd is not None
        and plan.estimated_loss_usd < 0
    ):
        plan.estimated_loss_usd = 0

    return plan