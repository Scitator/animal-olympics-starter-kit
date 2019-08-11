# [NeurIPS 2019: Animal-AI Olympics](http://animalaiolympics.com) <br/> [Catalyst](https://github.com/catalyst-team/catalyst) starter kit

## Environment setup.
(Taken from the official [repo](https://github.com/beyretb/AnimalAI-Olympics)).

The Animal-AI package works on Linux, Mac and Windows, as well as most Cloud providers. 
Note that for submission to the competition we only support linux-based Docker files.  
<!--, for cloud engines check out [this cloud documentation](documentation/cloud.md).-->

We recommend using a virtual environment specifically for the competition. You will need `python3.6` installed (we currently only support **python3.6**).

The main package is an API for interfacing with the Unity environment. 
It contains both a  [gym environment](https://github.com/openai/gym) as well as an extension of Unity's 
[ml-agents environments](https://github.com/Unity-Technologies/ml-agents/tree/master/ml-agents-envs). 
You can install it via pip:
    ```
    pip install animalai
    ```
    Or you can install it from the source, head to `animalai/` folder and run `pip install -e .`

Additionally download the environment for your system:

| OS | Environment link |
| --- | --- |
| Linux |  [download v1.0.0](https://www.doc.ic.ac.uk/~bb1010/animalAI/env_linux_v1.0.0.zip) |
| MacOS |  [download v1.0.0](https://www.doc.ic.ac.uk/~bb1010/animalAI/env_mac_v1.0.0.zip) |
| Windows | [download v1.0.0](https://www.doc.ic.ac.uk/~bb1010/animalAI/env_windows_v1.0.0.zip)  |

You can now unzip the content of the archive to the `assets` folder and you're ready to go! Make sure the executable 
`AnimalAI.*` is in `assets/`. On linux you may have to make the file executable by running `chmod +x assets/AnimalAI.x86_64`. 
Head over to [Quick Start Guide](documentation/quickstart.md) for a quick overview of how the environment works.


tl;dr
```bash
# system requirements
sudo apt-get install xvfb redis-server

# python requirements
conda create -n animal python=3.6 anaconda
source activate animal
pip install -r ./requirements.txt

# download and unzip env
```


## Catalyst.RL

To train agents on the Animal Olympics environment, we can run Catalyst.RL as usual.
```bash
# start db node
redis-server --port 12012

# start trainer node
export GPUS=""  # like GPUS="0" or GPUS="0,1" for multi-gpu training
CUDA_VISIBLE_DEVICES="$GPUS" catalyst-rl run-trainer --config ./configs/_exp_common.yml ./configs/ppo.yml

# start sampler node
CUDA_VISIBLE_DEVICES="" catalyst-rl run-samplers --config ./configs/_exp_common.yml ./configs/ppo.yml --sampler-id=1

# view tensorboard logs
CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs
```

For more information about Catalyst.RL you can check [official repo](https://github.com/catalyst-team/catalyst), [documentaiton](https://catalyst-team.github.io/catalyst/) and [examples](https://github.com/catalyst-team/catalyst/tree/master/examples/rl_gym).
