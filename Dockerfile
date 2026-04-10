# 1. Use a full Python image instead of 'slim' to ensure all tools are present
FROM python:3.9

# 2. Set the working directory
WORKDIR /app

# 3. Copy requirements first
COPY requirements.txt .

# 4. Install dependencies and FORCE them into the system path
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your project files
COPY . .

# 6. Set environment variable to ensure Python finds the installed packages
ENV PYTHONPATH=/app

# 7. Use the absolute path to run the module
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]