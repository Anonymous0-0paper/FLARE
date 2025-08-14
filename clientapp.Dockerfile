FROM flwr/clientapp:1.20.0
LABEL authors="leonkiss"

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r ./requirements.txt

COPY . /app

RUN pip install -e .

ENTRYPOINT ["flwr-clientapp"]