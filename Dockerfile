# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Only copy requirements first, for better cache
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the app code
COPY . /app

# Expose port
EXPOSE 80

# Run when container launches
CMD ["python", "-m", "app"]
