# Airflow 3 standalone with this repository installed, for the local UI.
# Build context is the repository root:  docker compose -f docker/airflow/docker-compose.yml up
FROM python:3.11-slim
ENV AIRFLOW_HOME=/opt/airflow \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__CORE__DAGS_FOLDER=/opt/repo/dags \
    PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 && rm -rf /var/lib/apt/lists/*
ARG AIRFLOW_VERSION=3.3.2
RUN pip install "apache-airflow==${AIRFLOW_VERSION}" \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.11.txt"
WORKDIR /opt/repo
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -e .
EXPOSE 8080
CMD ["airflow", "standalone"]
