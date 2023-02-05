# AAAI 2023: On the Challenges of using Reinforcement Learning in Precision Drug Dosing: Delay and Prolongedness


### Create and activate virtual environment
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
If you are using `simglucose` please cite the below:
1. The original original [UVa/Padova Simulator (2008 version)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4454102/)
2. `simglucose` [implemention by _Jinyu Xie_](https://github.com/jxx123/simglucose)

### Install paepomdp
```
git clone git@github.com:sumanabasu/On-the-Challenges-of-using-Reinforcement-Learning-in-Precision-Drug-Dosing-Delay-and-Prolongedness-.git
cd 'On-the-Challenges-of-using-Reinforcement-Learning-in-Precision-Drug-Dosing-Delay-and-Prolongedness'
pip install -e .
```

### Navigation tips
* The reward function and other utilities such as Wrappers to discretize actions are here:
`paepomdp/diabetes/helpers`


## Directory structure
```
├── On-the-Challenges-of-using-Reinforcement-Learning-in-Precision-Drug-Dosing-Delay-and-Prolongedness
    ├── Experiments       		# Trained models and results
        └── ...
    ├── paepomdp     
        └── algos
	    └── ...
	└── diabetes			# experiments on the modified simglucose environment
	    └── helpers
		└── rewards.py		# contains the zone_reward function introduced in the paper
		└── utils.py		# utility funcs used by the models
	    └── trainers
		└── ...
	    └── mains
		└── ...
├── setup.py            # To install paepomdp
```