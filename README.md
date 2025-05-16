# Inter-token Contrast (ICon)

[[Project Website]](https://anonymous.4open.science/w/ICon/)
[[Paper (Under Review)]]()
[[Data]](https://drive.google.com/drive/folders/16Z23gQSEiagQwN_gC7KdPvEJObEH3kaS?usp=sharing)

<img src="assets/icon.svg" alt="drawing" width="100%"/>

## 🔧 Installation
We recommend that users first create a Conda environment:
```bash
conda create -n icon_env python=3.10
conda activate icon_env
```
Then, install the Python package:
```
pip install git+https://github.com/HenryWJL/icon.git
```
This will automatically install all dependencies required to reproduce our experimental results in simulation. Note that running the RLBench environment requires **CoppeliaSim** to be installed. If you haven't installed CoppeliaSim yet, please follow the instructions [here](https://github.com/stepjam/RLBench?tab=readme-ov-file#install) to set it up.
 
## 💻 Training
### Downloading Dataset
We provide a new dataset spanning 8 manipulation tasks across 3 different robots from the RLBench and Robosuite benchmarks. To use our dataset, create a subdirectory at `data` in the project root and download the dataset from the web:
```bash
mkdir -p data
wget -P data TODO
``` 

### Running on a Device 
Now it’s time to give it a try! You can run `scripts/train.py` to train any algorithm on any task you like.
For example, to train a CNN-based diffusion policy on the *Open Box* task, simply run:
```bash
python scripts/train.py task=open_box algo=icon_diffusion_unet
```
This will automatically create a subdirectory at `outputs/TASK_NAME/ALGO_NAME/YYYY-MM-DD/HH-MM-SS`, where configuration files, log files, and checkpoints will be saved. If you want to run on a different device with a different seed, simply append the desired arguments to the original command:
```bash
python scripts/train.py task=open_box algo=icon_diffusion_unet train.device=cuda:0 train.seed=100
```

### 📐 Evaluating Pre-trained Checkpoints in Simulation
You can evaluate task success rate by running the following command: 
```bash
python scripts/eval.py -w clear -e rlbench -c @CHECKPOINT_PATH -s 100
```
This will rollout the pre-trained policy in the RLBench environment. If the robot successfully completes the task, or the running iteration exceeds the maximum rollout steps (100 in this situation), the program will automatically terminate. 
