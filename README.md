<div align="center">

# [Categorical Flow Maps](https://arxiv.org/abs/2602.12233)

_[Daan Roos*](https://mrroose.github.io/), [Oscar Davis*](https://olsdavis.github.io), [Floor Eijkelboom*](https://flooreijkelboom.github.io/),<br> [Michael Bronstein](https://www.cs.ox.ac.uk/people/michael.bronstein/), [Max Welling](https://amlab.science.uva.nl/people/MaxWelling/), [İsmail İlkan Ceylan](https://www.cs.ox.ac.uk/people/ismaililkan.ceylan/), [Luca Ambrogioni](https://www.artcogsys.com/team/luca), [Jan-Willem van de Meent](https://jwvdm.github.io/)_

Official implementation of the text experiments. :rocket:

[![arXiv](https://img.shields.io/badge/arXiv-2602.12233-red.svg)](https://arxiv.org/abs/2602.12233) ![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)

<img src="res/overview.png" width="80%">
</div>

## :question: About
This repository contains all the code for the text experiments from the Categorical Flow Maps paper. The main module of the code is located in [`semicat/models/semicat.py`](semicat/models/semicat.py) :brain:. The module is general and ready to accept many other data types. Text-specific code is to be found in [`semicat/models/textsemicat.py`](semicat/models/textsemicat.py) :pencil:.

## :gear: Running the code
1. Install the dependencies:
```sh
mamba env create -f environment.yaml
```

2. Activate the environment:
```sh
mamba activate semicat
```

3. Create a `.env` file containing the directory that will cache the processed LM1B data:
```
DATASET_CACHE_DIR=/the/dir/for/lm1b
```

4. Run the experiment you want! :boom: For example,
```sh
python -m semicat.train experiment=lm1b_dit trainer=gpu
```
For wandb logging, add `logger=wandb` as an argument.

## :bar_chart: Data

### Text8
To download the dataset, follow the steps in [github.com/andrew-cr/discrete_flow_models](https://github.com/andrew-cr/discrete_flow_models), placing the data in `./data/text8`.

### LM1B
LM1B is automatically downloaded into `DATASET_CACHE_DIR`, and then sequence-packed, etc. You can also run `python -m semicat.data.lm1b` separately in order to set up the data before launching your runs.

## :blue_book: Citation
To cite the paper or the code, please use the following:
```
@misc{roos2026categoricalflowmaps,
    title={Categorical Flow Maps}, 
    author={Daan Roos and Oscar Davis and Floor Eijkelboom and Michael Bronstein and Max Welling and İsmail İlkan Ceylan and Luca Ambrogioni and Jan-Willem van de Meent},
    year={2026},
    eprint={2602.12233},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2602.12233}, 
}
```
