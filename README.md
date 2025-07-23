# MixPE Simulator

This repository contains the code for the MixPE simulator based on DNNWeaver and BitFusion. The experimental results in paper "MixPE: Quantization and Hardware Co-design for Efficient LLM Inference" can be reproduced with the source code in this repo.

## Prerequisite

+ Ubuntu 22.04.1
+ Miniconda 25.1.1
+ Python 3.8.20
+ gcc 11.2.0

## Getting Started

```shell
$ # Environment.
$ conda create -n mixpe_sim python=3.8
$ conda activate mixpe_sim  
$ pip install -r  requirements.txt
$ # Cacti for the memory simulation.
$ git clone https://github.com/HewlettPackard/cacti ./bitfusion/sram/cacti/
$ make -C ./bitfusion/sram/cacti/
$ # Run end-to-end simulation for mixpe and other hardware simulators.
$ python ./run_mix.py
```

## Evaluation

The script `run_mix.py` generates statistic data and stores it in file `./result/mixpe_res.csv`.

In `./result/mixpe_res.csv`, Line 3 shows the **cycle**, i.e., Time, data that normalized with FP16. Line 7-10 shows the **energy** data that normalized with FP16.
