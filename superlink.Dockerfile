FROM nvcr.io/nvidia/pytorch:25.06-py3

WORKDIR /app

COPY link-node.requirements.txt /app

RUN pip install -r link-node.requirements.txt

EXPOSE 9091 9092 9093

ENTRYPOINT ["flower-superlink"]