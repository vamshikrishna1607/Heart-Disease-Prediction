FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DJANGO_USE_MANIFEST_STATIC=true \
    DJANGO_DEBUG=false \
    PYTHONUNBUFFERED=1

RUN python manage.py collectstatic --noinput

EXPOSE 8000

CMD ["gunicorn", "heart_disease_prediction.wsgi:application", "--bind", "0.0.0.0:8000"]
