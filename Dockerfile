FROM tiangolo/meinheld-gunicorn-flask:python3.7

LABEL maintainer="Agile Scientific"

RUN pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY ./ /app