FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (if any needed for reportlab/matplotlib)
# RUN apt-get update && apt-get install -y ...

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Make run script executable
RUN chmod +x run.sh

# Entrypoint
CMD ["./run.sh"]
