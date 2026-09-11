import os

from dotenv import load_dotenv


load_dotenv()


def get_int(
    name: str,
    default: int,
) -> int:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)

    except ValueError:
        raise RuntimeError(
            f"{name} must be an integer."
        )


def get_float(
    name: str,
    default: float,
) -> float:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return float(value)

    except ValueError:
        raise RuntimeError(
            f"{name} must be a number."
        )


class Settings:

    # -----------------------------------------------------
    # APPLICATION
    # -----------------------------------------------------

    APP_NAME = os.getenv(
        "APP_NAME",
        "Manufacturing Operations Agent API",
    )

    APP_VERSION = os.getenv(
        "APP_VERSION",
        "1.0.0",
    )

    APP_ENV = os.getenv(
        "APP_ENV",
        "development",
    )

    # -----------------------------------------------------
    # SECURITY
    # -----------------------------------------------------

    APP_API_KEY = os.getenv(
        "APP_API_KEY"
    )

    # -----------------------------------------------------
    # AI PROVIDER
    # -----------------------------------------------------

    OPENROUTER_API_KEY = os.getenv(
        "OPENROUTER_API_KEY"
    )

    OPENROUTER_BASE_URL = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )

    AI_MODEL = os.getenv(
        "AI_MODEL",
        "openrouter/free",
    )

    # -----------------------------------------------------
    # AGENT RELIABILITY
    # -----------------------------------------------------

    AGENT_TIMEOUT_SECONDS = get_int(
        "AGENT_TIMEOUT_SECONDS",
        30,
    )

    MAX_AGENT_ATTEMPTS = get_int(
        "MAX_AGENT_ATTEMPTS",
        2,
    )

    RETRY_DELAY_SECONDS = get_float(
        "RETRY_DELAY_SECONDS",
        1,
    )


settings = Settings()


# =========================================================
# REQUIRED SETTINGS VALIDATION
# =========================================================

if not settings.OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is missing."
    )

if not settings.APP_API_KEY:
    raise RuntimeError(
        "APP_API_KEY is missing."
    )

if settings.AGENT_TIMEOUT_SECONDS <= 0:
    raise RuntimeError(
        "AGENT_TIMEOUT_SECONDS must be greater than 0."
    )

if settings.MAX_AGENT_ATTEMPTS <= 0:
    raise RuntimeError(
        "MAX_AGENT_ATTEMPTS must be greater than 0."
    )

if settings.RETRY_DELAY_SECONDS < 0:
    raise RuntimeError(
        "RETRY_DELAY_SECONDS cannot be negative."
    )