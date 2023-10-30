# [AAAI 2023: On the Challenges of using Reinforcement Learning in Precision Drug Dosing: Delay and Prolongedness](https://arxiv.org/abs/2301.00512)

Disclaimer: Under Progress!

Short Video Link: https://youtu.be/YwhvZNyCxX8

## Create and activate virtual environment
```
conda create -n paepomdp_diabetes python=3.7 anaconda
conda activate paepomdp_diabetes
```

### Install pytorch 1.8.1
We have used `pytorch 1.8.1` in this project. Install the one compatible with your OS and CUDA version from here: 
https://pytorch.org/get-started/previous-versions/

### Install simglucose
We have made a few changes on the original simglucose environment. Changes are listed in the paper appendix. Install our version from here:
```
git clone git@github.com:sumanabasu/simglucose.git
cd simglucose
pip install -e .
```
If you are using `simglucose` please also cite the below:
1. The original [UVa/Padova Simulator (2008 version)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4454102/) paper
2. `simglucose` [implemention by _Jinyu Xie_](https://github.com/jxx123/simglucose)

### Install paepomdp
```
git clone git@github.com:sumanabasu/On-the-Challenges-of-using-Reinforcement-Learning-in-Precision-Drug-Dosing-Delay-and-Prolongedness-.git
cd 'On-the-Challenges-of-using-Reinforcement-Learning-in-Precision-Drug-Dosing-Delay-and-Prolongedness'
pip install -e .
```

## Training Models
To train an agent for the glucose control task, navigate to the trainers folder `paepomdp/diabetes/trainers` and run the training script for the specific model. \\
For example, to train the Effective DQN agent with the default hyperparameters, run
```
python effdqn_real_diabetes_trainer.py --patient_name adult#009
```
In the paper, we have reported results over 5 seeds. To run all of them, launch a slurm job array with the codes stored under `paepomdp/diabetes/mains/`.
For beginners, example slurm job array launchers can be found under `paepomdp/diabetes/launchers/`

The default environment in this paper is for `adult#009`. To change this, launch the code with the argument
`--patient_name`. Eg.:
```angular2html
python adrqn_diabetes_main.py --array_id=$SLURM_ARRAY_TASK_ID --patient_name=child#009
```


## Directory structure
```
├── On-the-Challenges-of-using-Reinforcement-Learning-in-Precision-Drug-Dosing-Delay-and-Prolongedness
    ├── Experiments       		# Trained models and results
        └── ...
    ├── paepomdp     
        └── algos			# RL agents
	    └── ...
	└── diabetes			# everything on the modified simglucose environment
	    └── helpers
		└── rewards.py		# contains the zone_reward function introduced in the paper
		└── utils.py		# utility funcs used by the env
	    └── trainers
		└── ...			# individual trainers can be considered as individual entry points to different models
	    └── mains
		└── ...
	└── MoveBlock			# Everything on the tabular toy environment			
├── setup.py            # To install paepomdp
```

## Citation
If you find our work interesting and use it in your research, please cite:

```bibtex
@article{basu2023paepomdp,
  title={On the Challenges of using Reinforcement Learning in Precision Drug Dosing: Delay and Prolongedness of Action Effects},
  author={Sumana Basu, Marc-André Legault, Adriana Romero-Soriano, Doina Precup},
  journal={Thirty-Seventh AAAI Conference on Artificial Intelligence},
  year={2023},
  url={https://arxiv.org/abs/2301.00512},
  arxiv={2301.00512},
}
```