# Pre/Post Processor
This repo is to run the pre/post processing (Tiling / Reconstruction) of JP2 files locally

### Prerequisites
  - Docker/Desktop
  - VSCode
  - VSCode Extensions
    - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
    - [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
    - [Remote Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

## How to run
  - In the dockfiler there are 2 separate sections 1 for gdal and 1 for opencv, uncomment either one based on the jupyter files you will be running (build & run commands are commented out for copy & paste purposes)
  - After building and running, attach vscode to container
  - Place image in the input folder for processing
  - Run the desired notebook file
