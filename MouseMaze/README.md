# Build docker environment 
docker build -t dirl .

# Run Code in Docker (Windows)
docker run -ti --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v D:\Work\Code\reinforcement_learning\MouseMaze:/mnt/MouseMaze -e DISPLAY=host.docker.internal:0.0 dirl /bin/bash