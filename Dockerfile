# Use the official Python image
FROM python:latest

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the project code into the container
COPY . .

# Expose Ray Dashboard port
EXPOSE 8265

# Run Ray
ENTRYPOINT ["ray", "start", "--head", "--dashboard-host=0.0.0.0"]
