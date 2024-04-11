FROM python:3.7-buster
EXPOSE 5000

RUN apt-get -y update && apt-get -y upgrade && apt-get install -y sqlite3 libsqlite3-dev

RUN mkdir -p /usr/src
RUN mkdir /data
WORKDIR /usr/src

RUN UWSGI_EMBED_PLUGINS="msgpack" pip install --no-cache-dir uwsgi
COPY requirements.txt /usr/src/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . /usr/src/
CMD uwsgi uwsgi.ini
