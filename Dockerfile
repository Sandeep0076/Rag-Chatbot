#  ╭──────────────────────────────────────────────────────────╮
#  │                        Base - Stage                      │
#  ╰──────────────────────────────────────────────────────────╯
FROM europe-west1-docker.pkg.dev/mgr-platform-prod-khsu/image-hub/dockerhub/python:3.11-slim AS base
ENV PYTHONUNBUFFERED=true

# adding non root user "worker", with no password, uid needed for workflow
RUN adduser \
    --disabled-password \
    --uid 4711 \
    worker
RUN mkdir -p /code /opt/poetry /nltk_data \
    && chown -R worker:worker /code /opt/poetry /nltk_data \
    && apt-get update
RUN apt-get install -y \
    # required for chat with image
    tesseract-ocr tesseract-ocr-deu \
    poppler-utils netcat-traditional \
    # requied for chat with doc(x) older formats
    antiword \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

#  ╭──────────────────────────────────────────────────────────╮
#  │                       Build - Stage                      │
#  ╰──────────────────────────────────────────────────────────╯
FROM base AS build

ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update && apt-get install build-essential -y  && apt-get install -y --no-install-recommends \
    curl gcc musl-dev build-essential\
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && curl -sSL https://install.python-poetry.org | python3 -

USER worker
WORKDIR /code
COPY --chown=nobody . ./

RUN python3 -m pip install --user nltk
ARG GITLAB_CI_TOKEN
RUN poetry config repositories.python-packages https://gitlab.com/api/v4/projects/33281928/packages/pypi/simple/
RUN poetry config http-basic.python-packages gitlab-ci-token ${GITLAB_CI_TOKEN} --no-interaction
RUN poetry install --no-interaction --only main
    # Clear the cache: it is mostly pip and poetry cache and the pipeline crashed without clearing it
RUN rm -rf /home/nobody/.cache/pypoetry/cache \
    && rm -rf /home/nobody/.cache/pypoetry/artifacts
    # Prepare nltk punkt data beforehand
RUN python -m nltk.downloader punkt -d /nltk_data

#  ╭──────────────────────────────────────────────────────────╮
#  │                     Runtime - Stage                      │
#  ╰──────────────────────────────────────────────────────────╯
FROM base AS runtime
ENV PATH="/opt/poetry/bin:code/.venv/bin:$PATH"

COPY --from=build /opt/poetry /opt/poetry
COPY --from=build --chown=worker:worker /code /code
COPY --from=build --chown=worker:worker /nltk_data /nltk_data

USER worker
RUN mkdir -p /code/chroma_db && chmod -R 777 /code/chroma_db

EXPOSE 8080

ENTRYPOINT ["poetry", "run"]
CMD [ "start" ]
