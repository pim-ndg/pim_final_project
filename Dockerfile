FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the application code
COPY . .

EXPOSE $PORT
HEALTHCHECK CMD curl --fail http://localhost:$PORT/_stcore/health
CMD streamlit run --server.port=${PORT:-8501} --server.address=0.0.0.0 main.py