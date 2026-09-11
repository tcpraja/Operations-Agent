from fastapi import (
    Header,
    HTTPException,
    status,
)

from app.config import settings


async def verify_api_key(
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
    ),
):

    if not x_api_key:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key.",
        )

    if x_api_key != settings.APP_API_KEY:

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return True