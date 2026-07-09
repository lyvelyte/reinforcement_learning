# Build docker environment 
docker build -t dirl .

# Run Code in Docker (Windows)
docker run -ti --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v D:\Work\Code\reinforcement_learning\MouseMaze:/mnt/MouseMaze -e DISPLAY=host.docker.internal:0.0 dirl /bin/bash

# Train a full-map DQN with the vectorized collector
conda run -n ml python MouseMaze/MouseAgent.py --episodes 50000 --no-dashboard --no-infer

# Run the deterministic validation, final-test, and stress suites on a checkpoint
conda run -n ml python MouseMaze/MouseAgent.py --benchmark --benchmark-episodes 2000 --no-infer

# Render the exact full-map planner instead of a learned policy
conda run -n ml python MouseMaze/MouseAgent.py --no-train --planner-fallback
