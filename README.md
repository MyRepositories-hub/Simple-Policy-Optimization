# This is our implementation of PPO and SPO

## Mujoco
Create Anaconda environment
```bash
conda create -n mujoco_py311 python=3.11 --yes
conda activate mujoco_py311
```

Install the mujoco requirements
```bash
cd mujoco
pip install -r requirements.txt
```

Choose the CUDA version on the official PyTorch website: [https://pytorch.org/](https://pytorch.org/)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining requirements
```bash
pip install gymnasium[mujoco]
```

Train
```bash
python main.py
```

## Atari
Create Anaconda environment
```bash
conda create -n atari_py311 python=3.11 --yes
conda activate atari_py311
```

Install the atari requirements
```bash
cd atari
pip install -r requirements.txt
```

Choose the CUDA version on the official PyTorch website: [https://pytorch.org/](https://pytorch.org/)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining requirements
```bash
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
pip install gymnasium[other]
```

Train
```bash
python main.py
```
