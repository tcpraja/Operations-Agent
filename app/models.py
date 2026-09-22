from typing import Literal


from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


class IncidentRequest(BaseModel):

    incident: str = Field(
        min_length=1,
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

        if len(cleaned.split()) < 2:
            raise ValueError(
                "Incident description must contain "
                "at least 2 meaningful words."
            )

        return cleaned


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list)
    session_id: str | None = None


class FileDownload(BaseModel):
    filename: str
    mime_type: str
    data_base64: str


class ChatResponse(BaseModel):
    reply: str
    image_svg: str | None = None
    analysis: dict | None = None
    pareto: dict | None = None
    web_images: list[dict] | None = None
    web_sources: list[dict] | None = None
    chart_image: str | None = None
    download: FileDownload | None = None


class UploadResponse(BaseModel):
    reply: str
    extracted_text: str | None = None
    download: FileDownload | None = None


class ExportRequest(BaseModel):
    format: Literal["docx", "pptx", "xlsx", "pdf"]
    analysis: dict


class DeleteConversationsRequest(BaseModel):
    session_ids: list[str] = Field(default_factory=list)


class RenameConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class ReplaceConversationRequest(BaseModel):
    messages: list[dict] = Field(default_factory=list)


class ChatDecision(BaseModel):
    action: Literal[
        "reply", "analyze", "pump_diagram", "cutter_diagram",
        "knowledge_pareto", "web_image", "web_cost", "fishbone_diagram",
        "calculate", "custom_chart", "generate_document",
    ]
    reply: str
    incident: str | None = None
    web_image_query: str | None = None
    document_format: Literal["docx", "pptx", "xlsx", "pdf"] | None = None
    document_title: str | None = None
    document_section_headings: list[str] = Field(default_factory=list)
    document_section_bullets: list[str] = Field(default_factory=list)
    fishbone_problem: str | None = None
    fishbone_machine: list[str] = Field(default_factory=list)
    fishbone_method: list[str] = Field(default_factory=list)
    fishbone_material: list[str] = Field(default_factory=list)
    fishbone_manpower: list[str] = Field(default_factory=list)
    fishbone_measurement: list[str] = Field(default_factory=list)
    fishbone_environment: list[str] = Field(default_factory=list)
    calculation_expression: str | None = None
    chart_type: Literal["trend", "pie"] | None = None
    chart_title: str | None = None
    chart_labels: list[str] = Field(default_factory=list)
    chart_values: list[float] = Field(default_factory=list)


class IncidentActionPlan(BaseModel):
    """
    Structured output schema the agent itself must fill in.
    Kept free of chart/image fields so the LLM is never
    asked to generate them.
    """

    summary: str
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    estimated_loss_usd: float
    likely_causes: list[str]
    immediate_actions: list[str]
    verification_checks: list[str]
    escalation_required: bool
    confidence: float
    knowledge_sources: list[str] = Field(default_factory=list)
    equipment_specifications: str = Field(
        default=(
            "Not enough information to determine "
            "equipment specifications."
        )
    )
    supplier_availability: list[str] = Field(
        default_factory=list
    )
    spare_parts_cost_estimate: str = Field(
        default=(
            "No cost information was found from "
            "available web sources."
        )
    )
    needs_clarification: bool = Field(default=False)
    clarification_question: str | None = Field(default=None)


class IncidentAnalysisResponse(IncidentActionPlan):
    """
    API response schema. Adds server-generated chart
    images on top of the agent's structured plan.
    """

    severity_dashboard_image: str | None = Field(default=None)
    failure_trend_image: str | None = Field(default=None)
    failure_count_image: str | None = Field(default=None)
    web_sources: list[dict] = Field(default_factory=list)
    pareto_analysis: dict | None = Field(default=None)
