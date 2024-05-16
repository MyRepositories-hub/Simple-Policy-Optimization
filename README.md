# SPO outperforms PPO in all environments when the network deepens (five random seeds):
![MuJoCo](https://github.com/MyRepositories-hub/Simple-Policy-Optimization/blob/main/draw_return_mujoco.png)

# Training
**The experimental environment is `gymnasium`, and you need to execute the following command to install the dependencies:**
## MuJoCo
### Installation
```bash
pip install gymnasium
pip install gymnasium[mujoco]
```
### Reminder
Please change the code from 
```python
self.add_overlay(bottomleft, "Solver iterations", str(self.data.solver_iter + 1))
```
to 
```python
self.add_overlay(bottomleft, "Solver iterations", str(self.data.solver_niter + 1))
```
in line 593 of the file path `venv\Lib\site-packages\gymnasium\envs\mujoco\mujoco_rendering.py` to resolve the error

### Running
```python
import gymnasium as gym

env = gym.make('Humanoid-v4', render_mode='human')
while True:
    s, _ = env.reset()
    done = False
    while not done:
        a = env.action_space.sample()
        s_next, r, dw, tr, info = env.step(a)
        done = (dw or tr)
```
## Atari
### Installation
```bash
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
```
### Running
```python
import gymnasium as gym

env = gym.make('ALE/Breakout-v5', render_mode='human')
while True:
    s, _ = env.reset()
    done = False
    while not done:
        a = env.action_space.sample()
        s_next, r, dw, tr, info = env.step(a)
        done = (dw or tr)
```
