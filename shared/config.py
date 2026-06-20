import os
from dotenv import load_dotenv

load_dotenv()

# Discord
DISCORD_BOT_TOKEN = os.environ["DISCORD_BOT_TOKEN"]
DISCORD_WEBHOOK_SECRET = os.getenv("DISCORD_WEBHOOK_SECRET", "")
SUPPORT_CHANNEL_ID = int(os.environ["SUPPORT_CHANNEL_ID"])
ANALYZER_INPUT_CHANNEL_ID = int(os.environ["ANALYZER_INPUT_CHANNEL_ID"])
KNOWLEDGE_LOG_CHANNEL_ID = int(os.environ["KNOWLEDGE_LOG_CHANNEL_ID"])
ANALYZER_OUTPUT_CHANNEL_ID = int(os.environ["ANALYZER_OUTPUT_CHANNEL_ID"])
ADMIN_USER_IDS = [int(x.strip()) for x in os.environ["ADMIN_USER_IDS"].split(",")]

# OpenRouter
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Model selection based on ENV
ENV = os.getenv("ENV", "development")
MODEL = (
    os.getenv("MODEL_DEV", "deepseek/deepseek-chat")
    if ENV == "development"
    else os.getenv("MODEL_PROD", "anthropic/claude-sonnet-4-6")
)

# Deepgram
DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]

# Postgres
# Keep .env as plain postgresql:// — works for both Alembic and AsyncPostgresSaver
# AsyncPostgresSaver needs plain postgresql://, NOT postgresql+psycopg://
_raw_url = os.environ["POSTGRES_URL"]
POSTGRES_URL = _raw_url.replace("postgresql+psycopg://", "postgresql://")

# SQLAlchemy async URL (if needed later for your own async DB queries)
POSTGRES_URL_ASYNC = POSTGRES_URL.replace("postgresql://", "postgresql+psycopg://")

# Rules
RULES_FILE = os.getenv("RULES_FILE", "data/rules.md")