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
We provide a new dataset spanning 8 manipulation tasks across 3 different robots from the RLBench and Robosuite benchmarks. To use our dataset, create a `data` subdirectory in the project root and download the dataset from the web:
```bash
mkdir -p data
wget -P data TODO
``` 

### ⏳ Running for Epochs
Now it's time to have a try! Run the following command to start training a new policy with seed 1 on GPU:
```bash
python scripts/train.py --config-name=clear.yaml task=close_door seed=1 device=cuda dataset_dir='data/close_door'
```
This will load configuration from `cross_embodiment/configs/workspaces/clear.yaml` and create a directory `outputs/$WORKSPACE_NAME/$TASK_NAME/YYYY-MM-DD/HH-MM-SS` where configuration files, logging files, and checkpoints are written to. For more details of model and training configuration, find them under `cross_embodiment/configs/workspaces`.

### 📐 Evaluating Pre-trained Checkpoints in Simulation
You can evaluate task success rate by running the following command: 
```bash
python scripts/eval.py -w clear -e rlbench -c @CHECKPOINT_PATH -s 100
```
This will rollout the pre-trained policy in the RLBench environment. If the robot successfully completes the task, or the running iteration exceeds the maximum rollout steps (100 in this situation), the program will automatically terminate. 
