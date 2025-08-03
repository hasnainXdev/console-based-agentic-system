from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel, function_tool, RunContextWrapper
from openai import AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables from .env file
load_dotenv()


# set the gemini key
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize OpenAI client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)


# ---------------- Context ----------------
class UserContext(BaseModel):
    name: str
    is_premium_user: bool
    issue_type: str
    
    
# ---------------- Tools ----------------
@function_tool
async def refund(ctx: RunContextWrapper) -> str:
    user_name = ctx.context.name
    return f"{user_name}, your refund has been initiated."


refund.is_enabled = lambda ctx, agent: ctx.context.is_premium_user

@function_tool
async def restart_service(ctx: RunContextWrapper) -> str:
    user_name = ctx.context.name
    return f"{user_name}, your service is now being restarted."


restart_service.is_enabled = lambda ctx, agent: ctx.context.issue_type == "technical"

@function_tool
async def general_info(ctx: RunContextWrapper) -> str:
    user_name = ctx.context.name
    return f"Hi {user_name}, here's some general info about our services."


# ---------------- Specialized Agents ----------------
billing_agent = Agent(
    name="BillingAgent",
   instructions="""
You are the BillingAgent.

If the user requests a refund and they are a premium user, always call the refund tool.

Do not respond with a message directly. Only use the refund tool to generate the response.
""",
    tools=[refund]
)

technical_agent = Agent(
    name="TechnicalAgent",
      instructions="""
You are the TechnicalAgent.

If the user requests a service restart and the issue type is technical, always call the restart_service tool.

Do not respond with a message directly. Only use the restart_service tool to generate the response.
""",
    tools=[restart_service]
)

general_agent = Agent(
    name="GeneralAgent",
    instructions="""
    You are the GeneralAgent.

    If the user requests general information, always call the general_info tool.

    Do not respond with a message directly. Only use the general_info tool to generate the response.
    """,
    tools=[general_info]
)

triage_agent = Agent(
    name="TriageAgent",
 instructions="""
You are a triage agent. Your only job is to decide which specialized agent should handle the user query.

- If the issue is about billing (e.g., refunds, payments), hand it off to billing_agent.
- If the issue is technical (e.g., errors, service issues), hand it off to technical_agent.
- If it's a general question or not specific, hand it off to general_agent.

Do not answer the query yourself. Only route it based on the user's issue_type and context.
""",
    handoffs=[billing_agent, technical_agent, general_agent],
)

# ---------------- CLI Runner ----------------
async def main():
    print("\nWelcome to the Support Agent System")
    name = input("Enter your name: ")
    is_premium_input = input("Are you a premium user? (yes/no): ").strip().lower()
    issue_type = input("What type of issue are you facing? (billing/technical/general): ").strip().lower()
    
    is_premium = is_premium_input == "yes"
    context = UserContext(name=name, is_premium_user=is_premium, issue_type=issue_type)

    user_query = input("\nPlease describe your issue: ")
    print("\nRouting your query...\n")

    result = await Runner.run(triage_agent, user_query, run_config=config, context=context)

    print("\nFinal Output:")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())