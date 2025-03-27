# Use a fresh Python image
FROM python:3.9

WORKDIR /app

# Copy dependencies first for caching
COPY app/requirements.txt .
RUN pip install --no-cache-dir --force-reinstall -r requirements.txt

# Set up NLTK data directory
RUN mkdir -p /usr/local/nltk_data
ENV NLTK_DATA="/usr/local/nltk_data"

# Download required NLTK resources
RUN python -c "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data'); nltk.download('stopwords', download_dir='/usr/local/nltk_data')"

# Copy the rest of the application
COPY . .

CMD ["python", "app/main.py"]
