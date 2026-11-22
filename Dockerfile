FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 COOKIE_SECURE=1 HOME=/home/retina
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 && rm -rf /var/lib/apt/lists/* && useradd --create-home --uid 1000 retina
WORKDIR /app
COPY requirements.txt requirements-space.txt ./
RUN pip install --no-cache-dir -r requirements-space.txt
COPY *.py *.json *.keras ./
COPY templates/ templates/
COPY static/ static/
RUN mkdir -p /app/instance && chown retina:retina /app/instance
USER retina
EXPOSE 7860
CMD ["gunicorn","--bind","0.0.0.0:7860","--workers","1","--threads","2","--timeout","300","space_wsgi:app"]
