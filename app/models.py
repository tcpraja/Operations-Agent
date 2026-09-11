from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


class IncidentRequest(BaseModel):

    incident: str = Field(
        min_length=10,
        max_length=2000,
    )

    downtime_minutes: float | None = Field(
        default=None,
        ge=0,
        le=100000,
    )

    loss_rate_usd_per_hour: float | None = Field(
        default=None,
        ge=0,
        le=1_000_000,
    )

    @field_validator("incident")
    @classmethod
    def validate_incident(
        cls,
        value: str,
    ):

        cleaned = value.strip()

        if not cleaned:
            raise ValueError(
                "Incident description cannot be empty."
            )

        words = cleaned.split()

        if len(words) < 3:
            raise ValueError(
                "Incident description must contain "
                "at least 3 meaningful words."
            )

        return cleaned


class IncidentActionPlan(BaseModel):
    summary: str
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    estimated_loss_usd: float
    likely_causes: list[str]
    immediate_actions: list[str]
    verification_checks: list[str]
    escalation_required: bool
    confidence: float
    knowledge_sources: list[str] = Field(default_factory=list)