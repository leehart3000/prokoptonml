# The dnabarcode API: a small image that serves a prebuilt bundle.
#
# Build from the repository root with the serving bundle in ./bundle (CI
# downloads it and checks its manifest first):
#   docker build -t dnabarcode-api .
# Cloud Run sets PORT; locally, docker run -p 8080:8080 dnabarcode-api

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PROKOPTON_BUNDLE=/srv/bundle

WORKDIR /srv

# The exact package versions the models were saved with come first, so
# this layer is reused while the bundle's requirements stay the same
COPY bundle/requirements.txt /srv/requirements.txt
RUN pip install -r /srv/requirements.txt

# Our package without its training dependencies (MLflow is not needed)
COPY pyproject.toml README.md LICENSE /srv/src-package/
COPY src /srv/src-package/src
RUN pip install --no-deps /srv/src-package && rm -rf /srv/src-package

COPY bundle /srv/bundle

# Run as an unprivileged user
RUN useradd --system --uid 10001 --no-create-home api
USER api

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn --factory prokopton.ml.serving.app:create_app_from_env --host 0.0.0.0 --port ${PORT:-8080}"]
