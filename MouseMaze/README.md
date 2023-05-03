# Build docker environment 
docker build -t dirl .

# Run Code in Docker (Windows)
docker run -ti --rm --gpus all -v C:\Users\alvey\Desktop:/mnt/desktop -e DISPLAY=host.docker.internal:0.0 dirl /bin/bash