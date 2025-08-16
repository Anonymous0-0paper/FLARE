FROM flwr/serverapp:1.20.0
LABEL authors="leonkiss"

WORKDIR /app

COPY . /app

ENTRYPOINT ["flwr-serverapp"]