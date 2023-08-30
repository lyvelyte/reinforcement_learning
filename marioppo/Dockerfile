FROM nvcr.io/nvidia/pytorch:23.04-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Chicago \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN \
    # Install various dependencies.
    apt-get -y update -qq --fix-missing && \
    apt-get -y install \
    build-essential \
    pkg-config \
    cmake

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the required packages
RUN python -m pip install --upgrade pip wheel==0.38.4
RUN pip install --no-cache-dir -r requirements.txt

# Set up environment variables for X11
ENV DISPLAY=:0

# Install required dependencies
RUN apt-get install -y \
    x11-apps \
    python-opengl \
    xterm

# Copy mario to working directory
COPY /ROMs /app/ROMs
COPY /src /app/src
COPY randomAgent.py .
RUN python -m retro.import /app/ROMs
