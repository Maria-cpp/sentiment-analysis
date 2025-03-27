# Use Python official base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy and install dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK resources
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('omw-1.4'); nltk.download('punkt_tab')"

# Copy the rest of the application
COPY . .

# Command to run the application
CMD ["python", "app/main.py"]
