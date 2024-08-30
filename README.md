# Failure Forecasting in Low Cost Sensors using Deep Time Series Models

This repository consists of the codebase for the implementation of the paper **Failure Forecasting in Low Cost Sensors using Deep Time Series Models**

## Contents
- [`datasets`](/datasets) directory consisting of the sensors data.
- [`models`](/models) directory consisting of the saved models.
- [`batches.py`](/batches.py) file containing custom batch generator.
- [`preprocess.py`](/preprocess.py) module for preprocessing the data.
- [`train.py`](/train.py) module to train model of choice on the data.
- [`test.py`](/test.py) module to test the trained model.
- [`utils.py`](/utils.py) module contating the utility functions.

## Requirements
- Requires `anaconda`

## Instructions 

- Create a python 3.10 environment using anaconda &rarr; 
```bash 
conda create env -n failurepred python=3.10
```
- Activate the environment &rarr;
```bash
conda activate failurepred
```
- Run the following command to install the dependencies &rarr;

```bash
pip install -r requirements.txt
```

- Use the following code to train the model &rarr;
```bash
python train.py -m <model_name> -t <test_type> -trb <train_balance_mode> -teb <test_balance_mode>
```
- Run the following code to get more details on the available options and default value &rarr;
```bash
python train.py --help
```
- Use the following code to test the model &rarr;
```bash
python test.py -m <model_name> -t <test_type> -trb <train_balance_mode> -teb <test_balance_mode>
```
- Run the following code to get more details on the available options and default value &rarr;
```bash
python test.py --help
```
