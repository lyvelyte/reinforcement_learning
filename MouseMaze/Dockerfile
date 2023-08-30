FROM nvcr.io/nvidia/pytorch:23.07-py3

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Set up environment variables for X11
ENV DISPLAY=:0

# Install required dependencies
RUN apt-get update && apt-get install -y \
    x11-apps \
    xterm