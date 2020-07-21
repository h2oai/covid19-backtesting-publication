import numpy as np
import pandas as pd
import datetime
import sys
import json
from datetime import timedelta
from joblib import Parallel, delayed
import multiprocessing
import time
from models import ZmodelSEIRD, ZmodelPG, ZmodelBaseline, EpidemicModelParams, EpidemicModel
from sklearn.metrics import mean_squared_error

def loss_rmsle(ytrue, pred):
    "RMSLE"
    assert ytrue.shape == pred.shape
    return np.sqrt(mean_squared_error( np.log(1 + ytrue).flatten(), np.log(1 + pred).flatten() ))

def single_bt_fit_seird(df, key, dates, known_params=None):
    res = []
    for reporting_date in dates:
        if known_params is not None:
            if reporting_date.strftime("%Y-%m-%d") in known_params:
                if 'SEIRD' in known_params[reporting_date.strftime("%Y-%m-%d")]:
                    if key in known_params[reporting_date.strftime("%Y-%m-%d")]['SEIRD']:
                        continue

        df_short = df[(df['date'] <= reporting_date) & (df['key'] == key)]
        VAL_LEN = df['days'].max() - df_short['days'].max()
        if VAL_LEN < 1:
            continue
        params = EpidemicModelParams()
        model = EpidemicModel(params).fit(df_short, workers=1, maxiter=1000)
        res.append({'model': 'SEIRD', 'date':reporting_date.strftime("%Y-%m-%d"),
                       'key':key, 'params':model.best_params_dict()})
    return res

def fit_seird():
    # Read the input data
    df = pd.read_csv('files/covid_train_country_june_22.csv')\
            [['date', 'key', 'confirmed', 'deaths', 'recovered', 'confirmed_new', 'active']]
    df['date'] = df['date'].apply(lambda x: (datetime.datetime.strptime(x, '%m/%d/%y')))
    df['days'] = (df['date'].dt.date - df['date'].dt.date.min()).dt.days

    seird_params = {}

    # Run backtesting
    reporting_dates = [df['date'].min() + timedelta(days=30+i) for i in range(df['days'].max())]
    res = Parallel(n_jobs=8)(delayed(single_bt_fit_seird)\
            (df, key, reporting_dates, seird_params) for key in df['key'].unique())

    # Extract parameters
    res_params = [item for sublist in res for item in sublist if item is not None]

    # Restructure the list of parameters
    for z in res_params:
        if z['date'] not in seird_params:
            seird_params[z['date']] = {}
        if z['model'] not in seird_params[z['date']]:
            seird_params[z['date']][z['model']] = {}
        seird_params[z['date']][z['model']][z['key']] = z['params']

    # Save the results
    with open('files/backtesting/bt_seird_params.json', 'w') as fout:
        json.dump(seird_params, fout)

    return seird_params
        
def single_bt_fit_pg(df, target_name, reporting_date, model_type, params):
    
    MIN_DATE = df['date'].min()
    AREAS = np.sort(df['key'].unique())
    N_TRAIN = df[df['date'] <= reporting_date]['days'].max() + 1
    
    VAL_LEN = df['days'].max() - N_TRAIN + 1
    if VAL_LEN < 1:
        return None
    
    df_p = df.pivot(index='key', columns='days', values=target_name).sort_index().values
    model = model_type().opt(df_p[:, :N_TRAIN], params['loss_fun'],
                             valid_horizon=params['optim_horizon'], max_trials=params['max_trials'])
    return {'model': model.__doc__, 'date':reporting_date.strftime("%Y-%m-%d"),
                 'target':target_name, 'params':model.params}

def fit_pg():
    # Read the input data
    df = pd.read_csv('files/covid_train_country_june_22.csv')\
            [['date', 'key', 'confirmed', 'deaths', 'recovered', 'confirmed_new', 'active']]
    df['date'] = df['date'].apply(lambda x: (datetime.datetime.strptime(x, '%m/%d/%y')))
    df['days'] = (df['date'].dt.date - df['date'].dt.date.min()).dt.days

    pg_params = {}
        
    params = {'loss_fun':loss_rmsle, 'optim_horizon':21, 'max_trials':300}
    reporting_dates = [df['date'].min() + timedelta(days=30+i) for i in range(df['days'].max())]
    
    # Run backtesting
    res = Parallel(n_jobs=8)(delayed(single_bt_fit_pg)\
        (df, target, reporting_date, ZmodelPG, params)
        for reporting_date in reporting_dates \
        for target in ['confirmed', 'deaths', 'recovered'] \
        )

    # Extract parameters
    res_params = [item for item in res if item is not None]

    # Restructure the list of parameters
    for z in res_params:
        if z['date'] not in pg_params:
            pg_params[z['date']] = {}
        if z['model'] not in pg_params[z['date']]:
            pg_params[z['date']][z['model']] = {}
        pg_params[z['date']][z['model']][z['target']] = z['params']

    # Save the results
    with open('files/backtesting/bt_pg_params', 'w') as fout:
        json.dump(pg_params, fout)

    return pg_params

if __name__ == "__main__":
    start_time = time.time()
    fit_seird()
    fit_pg()
    print(f"Elapsed {time.time() - start_time : 8.5f} s")


