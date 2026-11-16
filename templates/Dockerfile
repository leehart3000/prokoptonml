# A product's API image: the ProkoptonML platform from PyPI, serving the
# product's prebuilt bundle. Copy this file to the product's repository as
# Dockerfile, and set the two ARG defaults below (the platform version must
# be the one the product pins in pixi.toml).
#
# The deploy workflow (deploy-cloud-run.yml) downloads the bundle into
# ./bundle, checks it against its manifest and builds from the repository
# root. To build by hand, with the bundle in ./bundle:
#   docker build -t my-api .
# Cloud Run sets PORT; locally, docker run -p 8080:8080 my-api

FROM python:3.13-slim

# The platform version, pinned exactly (prokopton-ml 0.2.0a4 or later, for
# the deploy workflow's known-answer checks)
ARG PROKOPTON_ML_VERSION=0.2.0a7
# The name shown in the API docs
ARG API_TITLE="My product API"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PROKOPTON_BUNDLE=/srv/bundle \
    PROKOPTON_API_TITLE="${API_TITLE}"

WORKDIR /srv

# The exact package versions the models were saved with come first, so
# this layer is reused while the bundle's requirements stay the same
COPY bundle/requirements.txt /srv/requirements.txt
RUN pip install -r /srv/requirements.txt

# The platform from PyPI, without its training dependencies (MLflow is not
# needed to serve)
RUN pip install --no-deps "prokopton-ml==${PROKOPTON_ML_VERSION}"

# A product with its own serving steps or model kinds (installed packages
# with prokopton.ml entry points) installs its package here, also with
# --no-deps, for example:
#   COPY dist/my_product-1.0.0-py3-none-any.whl /tmp/
#   RUN pip install --no-deps /tmp/my_product-1.0.0-py3-none-any.whl

COPY bundle /srv/bundle

# Run as an unprivileged user
RUN useradd --system --uid 10001 --no-create-home api
USER api

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn --factory prokopton.ml.serving.app:create_app_from_env --host 0.0.0.0 --port ${PORT:-8080}"]
