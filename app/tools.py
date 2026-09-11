from agents import function_tool


# =========================================================
# PURE PYTHON BUSINESS LOGIC
# Easy to test without an LLM
# =========================================================

def calculate_loss(
    downtime_minutes: float,
    loss_rate_usd_per_hour: float,
) -> dict:

    downtime_hours = downtime_minutes / 60

    loss = downtime_hours * loss_rate_usd_per_hour

    return {
        "downtime_hours": round(downtime_hours, 2),
        "estimated_loss_usd": round(loss, 2),
    }


def get_playbook(equipment: str) -> dict:

    playbooks = {
        "cutter": [
            "Inspect cutter blade condition",
            "Check cutter bearing temperature",
            "Check vibration",
            "Check cutter alignment",
            "Check motor current",
            "Review recent trip alarms",
        ],
        "pump": [
            "Check suction pressure",
            "Check discharge pressure",
            "Inspect mechanical seal",
            "Check motor current",
            "Check vibration",
        ],
        "compressor": [
            "Check discharge temperature",
            "Check suction pressure",
            "Check lubrication system",
            "Check vibration",
            "Review trip alarms",
        ],
    }

    equipment_lower = equipment.lower()

    for key, checks in playbooks.items():

        if key in equipment_lower:

            return {
                "equipment": key,
                "checks": checks,
            }

    return {
        "equipment": equipment,
        "checks": [
            "Inspect equipment condition",
            "Review alarm history",
            "Review maintenance history",
            "Check process parameters",
        ],
    }


# =========================================================
# AGENT TOOLS
# These wrappers expose the Python functions to the LLM
# =========================================================

@function_tool
def calculate_production_loss(
    downtime_minutes: float,
    loss_rate_usd_per_hour: float,
) -> dict:
    """Calculate production loss caused by equipment downtime."""

    print("\n[TOOL CALLED] calculate_production_loss")

    result = calculate_loss(
        downtime_minutes,
        loss_rate_usd_per_hour,
    )

    print(f"[TOOL RESULT] {result}\n")

    return result


@function_tool
def get_equipment_playbook(
    equipment: str,
) -> dict:
    """Get standard troubleshooting checks for manufacturing equipment."""

    print("\n[TOOL CALLED] get_equipment_playbook")
    print(f"Equipment requested: {equipment}")

    result = get_playbook(equipment)

    print(f"[TOOL RESULT] {result}\n")

    return result