FROM python:3.7.13-slim-bullseye

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libffi-dev gcc-9 g++-9 && \
    rm -rf /var/lib/apt/lists/*

ENV CXX=/usr/bin/g++-9
ENV CC=/usr/bin/gcc-9

WORKDIR /survival-analysis
COPY requirements.txt .
RUN pip install --no-cache -r requirements.txt
COPY . ./
# ENTRYPOINT ["python", "churn/main.py"]
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]