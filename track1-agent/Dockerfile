FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entrypoint and source code
COPY agent.py ./
COPY src/ ./src/

# Run the agent
CMD ["python3", "agent.py"]
