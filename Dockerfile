# Use the official Ubuntu 20.04 as a base image
FROM ubuntu:20.04

# Set environment variable to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /usr/src/app

# Install necessary packages and Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get clean

# Copy local files to the container
COPY main.py /usr/src/app
COPY mouseLFP.mat /usr/src/app
COPY szkim_library /usr/src/app/szkim_library
COPY requirements.txt /usr/src/app

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Install szkim_library package in editable mode
RUN pip3 install -e .

# Run main.py to generate the PNG files
CMD ["python3", "main.py"]
