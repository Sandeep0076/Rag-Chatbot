import uvicorn
from fastapi import FastAPI
from fastapi.params import Query
from starlette_exporter import PrometheusMiddleware, handle_metrics

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
    return {"name": __name__, "version": __version__}


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("example_fast_api.app:app", host="0.0.0.0", port=8080, reload=False)
