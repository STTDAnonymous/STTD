This is a Pytorch implementation of the following paper: 

# Papers
#### [🧩] Spatiotemporal Graph Neural Networks with Uncertainty Quantification for Traffic Incident Risk Prediction

# The Code

## Requirements

Following is the suggested way to install the dependencies:

    conda install --file STTD.yml

Note that ``pytorch >=1.10``.

## Folder Structure

```tex
└── code-and-data
    ├── cta_data_only10                 # CDPSAMP10 Dataset as an example
    ├── ny_data_only10                  # SLDSAMP10 Dataset as an example
    ├── ny_data_full_5min               # SLD_5min Dataset as an example
    ├── ny_data_full_15min              # SLD_15min Dataset as an example
    ├── ny_data_full_60min              # SLD_60min Dataset as an example
    ├── main_gau.py                     # STG Model
    ├── main_stnb.py                    # STNB Model
    ├── main_trunnorm.py                # STN Model
    ├── main_zero_NB.py                 # STZINB Model
    ├── main_tweedie.py                 # Tweedie Model (STTD, STP, STGM, STIG)
    ├── main_zitd.py                    # ZI-Tweedie Model
    ├── model.py                        # The core source code of our model
    ├── utils.py                        # Defination of auxiliary functions for running
    ├── STTD.yml                        # The python environment needed for STTD
    ├── pth                             # Best model save path
    └── README.md                       # This document
```

## Datasets

Download datasets from [ZhuangDingyi/STZINB: Source code of implementing spatial-temporal zero-inflated negative binomial network for trip demand prediction (github.com)](https://github.com/ZhuangDingyi/STZINB)

For London traffic risk dataset, please contact us for more details. 

## Configuration

Important parameters in the configuration are as follows :

```tex
nhid = 42                               # The hidden unit
weight_dacay = 1e-4                     # Weight decay
learning_rate = 1e-3                    # Learning rate
drop_out = 0.2                          # Dropout rate					 
```


##  Train and Test

Run *python main_{method}.py* to train and evaluate the model and generate model prediction .Remember to replace the corresponding data files and output files.

