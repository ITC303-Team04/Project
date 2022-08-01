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
  - In terminal navigate to this directory.
  - Build the image from the Dockerfile by running `docker built -t processing .` This may take a few mins
  - Once built, run the container from the image by `docker run -it processing`. You should see the prompt change to something similar to `root@e5487e324206:/# `
  - Once the container is running, open VSCode and in the bottom right click `Open a Remote Window` and select `Attach to Running Container...`
  - There should be a container with a random name followed by processing: `/<random_name> processing`, select that container
  - VScode will restart inside the container, if prompted to install kernel click yes and select `Install in Container`
  - Open the `to_tiles.ipynb` and in the top right click the kernel button and ensure the kernel is the Suggest Kernel

  After this the project should be ready to run, at the top select |> Run All 
