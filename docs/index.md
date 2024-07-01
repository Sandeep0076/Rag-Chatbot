# rag-pdf-api

## Description
RAG PDF API

## Links
This is an FastAPI project.
See https://fastapi.tiangolo.com/

## How to run locally:

1. Clone the repository

2. Create a [personal access token](https://gitlab.com/-/profile/personal_access_tokens).

3. Configure the remote repository
```
poetry config repositories.python-packages https://gitlab.com/api/v4/projects/33281928/packages/pypi/simple/
```

4. Configure your token
```
poetry config http-basic.python-packages <gitlab-token-name> <gitlab-token>
```

5. Install pre-commit
```
pre-commit install
```

6. Install the package:  
```
make install
```

7. Run project locally
```
make serve
```

8. Test project locally
```
make serve # starts application at http://127.0.0.1:8080

make e2e # create request to local api endpoint and prints response
```

9. Run Tests
```
make test
```
