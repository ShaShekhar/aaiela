## Installation:

**Prerequisites:**

- **Python >= 3.9**
- **PyTorch >= 1.8 with CUDA 11.x or 12.x support:**
  Follow the instructions on [official PyTorch website!](https://pytorch.org/)

**Steps:**

```bash
$ git clone https://github.com/shashekhar/aaiela.git
$ cd aaiela
$ pip install -r requirements.txt
$ # pip install --force-reinstall ctranslate2==3.24.0 # for cuda 11.8
$ cd models
$ python -m pip install -e detectron2
```

Follow [detectron2 install instruction](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

Download sd-v1.5-inpaint and clip 'ViT-L-14' model

```bash
$ cd .. # inside aaiela directory
$ wget -P weights https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt
$ wget -P weights https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
$ python app.py # tail -f /tmp/app.log
```

**Conda Installation:**

```bash
$ conda create -n aaiela python=3.9
$ conda activate aaiela
$ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia # or 11.8
$ git clone https://github.com/shashekhar/aaiela.git
$ cd aaiela
$ pip install -r requirements.txt
$ cd models
$ python -m pip install -e detectron2
$ cd ..
$ python -m tests.test_detectron
$ # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/anaconda3/envs/aaiela/lib/python3.9/site-packages/torch/lib
$ python -m tests.test_whisper
$ python app.py # start a tmux session
$ tail -f /tmp/app.log
```

**docker Installation:**

- The official [Dockerfile](Dockerfile) installs with a few simple commands.

```bash
$ git clone https://github.com/shashekhar/aaiela.git
$ cd aaiela
$ docker build -t aaiela_conda --network=host --build-arg CUDA_VERSION=12.1 . # or 11.8
$ docker run --gpus all -it --rm -v weights:/app/weights -p 5000:5000 aaiela_conda
$ conda activate aaiela
$ python app.py # start a tmux session
```
