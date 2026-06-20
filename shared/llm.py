from langchain_openai import ChatOpenAI
from shared.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL


def get_llm(temperature: float = 0.3, model_override: str = None) -> ChatOpenAI:
    """
    Returns a LangChain ChatOpenAI client pointed at OpenRouter.

    model_override: force a specific model regardless of ENV setting.
                    Used for cheap evaluation calls (deepseek) even in prod.
    """
    return ChatOpenAI(
        model=model_override or MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=temperature,
        default_headers={
            "HTTP-Referer": "https://your-company.com",
            "X-Title": "Discord Automation",
        },
    )