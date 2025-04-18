# Use official Python base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all files to the container
COPY . /app

# Copy and install dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords

# Expose the Chainlit default port
EXPOSE 8000

# Run Chainlit inside the container
CMD ["chainlit", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
