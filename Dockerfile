FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN python bootstrap/train_and_register.py
CMD ["uvicorn", "app.main:app","--host", "0.0.0.0", "--port", "9001"]