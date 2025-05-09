<h1 align="center">
	Simple Policy Optimization<br>
</h1>

<p align="center">
  Zhengpeng Xie*, Qiang Zhang*, Fan Yang*, Marco Hutter, Renjing Xu
</p>

Accepted to <i style="color: black; display: inline;"><b>International Conference on Machine Learning (ICML 2025)</b></i> | [arXiv](https://arxiv.org/abs/2401.16025)<br>





# Installation
## MuJoCo
Create Anaconda environment:
```bash
conda create -n mujoco_py311 python=3.11 --yes
conda activate mujoco_py311
```

Install the mujoco requirements:
```bash
cd mujoco
pip install -r requirements.txt
```

Choose the CUDA version on the official PyTorch website: https://pytorch.org/
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining requirements:
```bash
pip install gymnasium[mujoco]
```

Start training:
```bash
python main.py
```

## Atari
Create Anaconda environment:
```bash
conda create -n atari_py311 python=3.11 --yes
conda activate atari_py311
```

Install the atari requirements:
```bash
cd atari
pip install -r requirements.txt
```

Choose the CUDA version on the official PyTorch website: [https://pytorch.org/](https://pytorch.org/)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining requirements:
```bash
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
pip install gymnasium[other]
```

Start training:
```bash
python main.py
```

# Acknowledgement
Our code is mainly based on [cleanrl](https://github.com/vwxyzjn/cleanrl), many thanks to their efforts.
