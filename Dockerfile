FROM python:slim

WORKDIR /app/src

RUN pip install --no-cache-dir \
    transformers \
    torch \
    accelerate \
    ;
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT [ "python", "script.py" ]
