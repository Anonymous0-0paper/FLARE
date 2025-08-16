FROM nvcr.io/nvidia/pytorch:25.06-py3

WORKDIR /app

COPY link-node.requirements.txt /app

RUN pip install -r link-node.requirements.txt

ENTRYPOINT ["flower-supernode"]