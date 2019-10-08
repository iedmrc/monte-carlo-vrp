FROM python:3
ENV PYTHONUNBUFFERED definitely
WORKDIR /usr/src/app

COPY requirements.txt ./
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y libgmp-dev libmpfr-dev libmpc-dev && apt-get autoclean

RUN pip install --no-cache-dir -r requirements.txt


ENTRYPOINT [ "python", "./vrp.py" ]