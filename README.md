# sentiment-analysis
# Create main project folders
mkdir app data

# Navigate into the 'app' folder
cd app

# Create Python script files
touch main.py preprocess.py model.py requirements.txt

# Go back to the main directory
cd ..

# Create dataset file inside 'data' folder
touch data/dataset.csv

# Create Docker and config files
touch Dockerfile docker-compose.yml .gitignore README.md

#RUN Command
docker-compose up --build

