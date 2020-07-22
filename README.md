# Backtesting the predictability of COVID-19

This repository contains the relevant code to replicate the main results of our paper "Backtesting the predictability of COVID-19"

## Running backtesting

To run all historical fits of SEIRD and Power Growth models, run
```
python fit.py
```

## Generating figures

To generate the figures of Section 4 Experiments of the article, run the ```paper_plots.ipynb``` notebook accordingly.
It uses the stored backtested parameters from the respective json files in the ```files``` folder.
