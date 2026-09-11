from app.tools import (
    calculate_loss,
    get_playbook,
)


def test_loss_calculation():

    result = calculate_loss(
        downtime_minutes=120,
        loss_rate_usd_per_hour=500,
    )

    assert result["downtime_hours"] == 2.0
    assert result["estimated_loss_usd"] == 1000.0


def test_cutter_playbook():

    result = get_playbook(
        "PET Line 6 cutter"
    )

    assert result["equipment"] == "cutter"

    assert "Inspect cutter blade condition" in result["checks"]

    assert "Check vibration" in result["checks"]