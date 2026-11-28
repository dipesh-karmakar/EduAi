# Description: Builds and runs the Python tutoring API.
# Author: Dipesh Karmakar <dipesh.karmakar@gmail.com>
# Modifications:
# - 130fab8: Added the initial engine container build.

FROM python:3.12-slim

WORKDIR /app

# Install API dependencies separately to reuse the layer when application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 9000

# Run the OpenAI-compatible FastAPI service.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
