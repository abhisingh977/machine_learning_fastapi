FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9


COPY src/model /src/model
COPY src/required_data /src/required_data
COPY requirements.txt /src/requirements.txt
COPY src/app/ /src/app/
COPY src/main.py /src/main.py

WORKDIR /src
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt


# Expose the port
EXPOSE 1313

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1313"]