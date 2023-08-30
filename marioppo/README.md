# Build docker environment 
docker build -t dirl .

# Run Code in Docker (Windows)
docker run -ti --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v C:\Users\alvey\Desktop:/mnt/desktop -e DISPLAY=host.docker.internal:0.0 dirl /bin/bash

# Run random agent
python randomAgent.py

# Interactive Script 
python -m retro.examples.interactive --game SuperMarioBros-Nes