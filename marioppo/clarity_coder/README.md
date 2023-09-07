# Build docker environment 
docker build -t mario_01 .

# Run Code in Docker (Windows)
docker run -ti --rm --gpus all --ipc=host -v D:\Work\Code\reinforcement_learning\marioppo\clarity_coder:/app/clarity_coder -p 6006:6006 -e DISPLAY=host.docker.internal:0.0 mario_01 /bin/bash

# Run random agent
python RandomAgent.py

# Train agent
python Train.py

# Test agent
python Run.py

# Interactive Script 
python -m retro.examples.interactive --game SuperMarioBros-Nes