FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9


COPY app/model /app/model
COPY app/required_data /app/required_data
COPY requirements.txt /app/requirements.txt
COPY app/__init__.py /app/__init__.py
COPY app/function.py /app/function.py
COPY app/constants.py /app/constants.py
COPY app/schemas.py /app/schemas.py
COPY app/main.py /app/main.py
COPY app/tests /app/tests

WORKDIR /app
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

RUN pytest tests/

# Expose the port
EXPOSE 1313

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1313"]