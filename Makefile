PROJECT_NAME=rag-pdf-api

install:
	@poetry update
	@poetry install

serve:
	@poetry run start --reload

test:
	@poetry run pytest

lint:
	@poetry run black .
	@poetry run isort . --check --diff --multi-line 3

fixlint:
	@poetry run isort . --multi-line 3

pre-commit:
	@poetry run pre-commit run --all-files

e2e:
	@curl http://localhost:8080/
