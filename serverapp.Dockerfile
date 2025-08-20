FROM flwr/serverapp:1.20.0

WORKDIR /app

COPY . /app

ENTRYPOINT ["flwr-serverapp"]
