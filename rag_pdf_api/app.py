import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.params import Query
from starlette_exporter import PrometheusMiddleware, handle_metrics

from configs.app_config import Config
from rag_pdf_api.chatbot.chatbot_creator import Chatbot

configs = Config()
chatbot = Chatbot(configs)

from rag_pdf_api import __name__, __version__

app = FastAPI()

# expose prometheus metrics at /metrics rest endpoint
app.add_middleware(
    PrometheusMiddleware,
    group_paths=True,
    app_name="api",
    filter_unhandled_paths=True,
    # do not report on health and metrics endpoint in order to keep the
    # size of the metrics low
    skip_paths=["/health", "/metrics"],
)
app.add_route("/metrics", handle_metrics)


@app.get("/")
async def root(
    message: str = Query("Hello World", max_length=10),
):
    return {
        "message": message,
    }


@app.get("/health")
async def health():
    """
    Shows application health information.
    In the future this could do some actual checks.
    """
    return {"status": "up"}


@app.get("/info")
async def info():
    """
    Displays arbitrary application info.
    This could be the service name and some build info like
    e.g. the commit hash or the build time.
    """
    return {
        "title": configs.chatbot.title,
        "description": configs.chatbot.description,
        "info_text": configs.chatbot.info_text,
    }


@app.get("/pdf/answer")
async def answer(query: Query):
    try:
        if query.llm_only:
            response = chatbot.get_llm_answer(query.text)
        else:
            response = chatbot.get_answer(query.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("rag_pdf_api.app:app", host="0.0.0.0", port=8080, reload=False)
