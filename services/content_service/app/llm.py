import httpx
from openai import AsyncOpenAI
from .config import settings

client = AsyncOpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    http_client=httpx.AsyncClient(verify=False),
)


async def chat(system: str, user: str) -> str:
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content
