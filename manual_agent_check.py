import asyncio
import os

from dotenv import load_dotenv


# Load .env
load_dotenv()


if not os.getenv("OPENROUTER_API_KEY"):
    raise RuntimeError(
        "OPENROUTER_API_KEY was not found."
    )


from agents import Runner

from app.agent import operations_agent


async def main():

    print("\nStarting Manufacturing Agent...\n")

    result = await Runner.run(

        operations_agent,

        """
PET Line 6 cutter tripped four times today.

Operators noticed abnormal vibration.

Downtime was 120 minutes.

Estimated production loss rate is
500 USD per hour.

Analyze this incident.
"""
    )

    print("\n--- AGENT OUTPUT ---\n")

    print(result.final_output)


if __name__ == "__main__":

    asyncio.run(main())