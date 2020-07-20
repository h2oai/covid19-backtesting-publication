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
from defaults import *
from sklearn.metrics import mean_squared_error
import os
import boto3

def loss_mae(ytrue, pred):
    "MAE"
    assert ytrue.shape == pred.shape
    return np.mean(np.abs(pred.flatten()-ytrue.flatten()))

def loss_mape(ytrue, pred):
    "MAPE"
    assert ytrue.shape == pred.shape
    y = ytrue.flatten()
    pred = pred.flatten()[y > 0]
    y = y[y>0]
    return np.mean(np.abs((pred-y) / y))

def loss_rmsle(ytrue, pred):
    "RMSLE"
    assert ytrue.shape == pred.shape
    return np.sqrt(mean_squared_error( np.log(1 + ytrue).flatten(), np.log(1 + pred).flatten() ))


def single_bt_fit_seird(df, key, dates, known_params=None):

    res = []
    for reporting_date in dates:

        if known_params is not None:
            if reporting_date.strftime("%Y-%m-%d") in known_params:
                if key in known_params[reporting_date.strftime("%Y-%m-%d")]:
                    continue

        df_short = df[(df['date'] <= reporting_date) & (df['key'] == key)]
        VAL_LEN = df['days'].max() - df_short['days'].max()
        if VAL_LEN < 1:
            continue

        params = EpidemicModelParams()
        model = EpidemicModel(params).fit(df_short, workers=1, maxiter=1000)

        res.append({'date':reporting_date.strftime("%Y-%m-%d"),
                       'key':key, 'params':model.best_params_dict()})
    return res

def fit_seird(in_file=seird_params_file, out_file=seird_params_file):
    
    # Read the input data
    df = pd.read_csv(country_file)\
            [['date', 'key', 'confirmed', 'deaths', 'recovered', 'confirmed_new', 'active']]
    df['date'] = df['date'].apply(lambda x: (datetime.datetime.strptime(x, '%m/%d/%y')))
    df['days'] = (df['date'].dt.date - df['date'].dt.date.min()).dt.days
    #population = pd.read_csv(bt_file_population)

    # Check if a file with fitted parameters exists and load it
    try:
        with open(in_file, "r") as read_file:
            seird_params = json.load(read_file)
    except:
        seird_params = {}
        
    reporting_dates = [df['date'].min() + timedelta(days=30+i) for i in range(df['days'].max())]
    #reporting_dates = [df['date'].min() + timedelta(days=30+i) for i in range(2)]
    
    # Run backtesting
    res = Parallel(n_jobs=N_JOBS)(delayed(single_bt_fit_seird)\
            (df, key, reporting_dates, seird_params) for key in df['key'].unique())

    # Extract parameters
    res_params = [item for sublist in res for item in sublist if item is not None]

    # Restructure the list of parameters
    for z in res_params:
        if z['date'] not in seird_params:
            seird_params[z['date']] = {}
        seird_params[z['date']][z['key']] = z['params']

    # Save the results
    if out_file is not None:
        with open(out_file, 'w') as fout:
            json.dump(seird_params, fout)

    return seird_params
        
def single_bt_fit_pg(df, reporting_date, params, known_params=None):
    if known_params is not None:
        if reporting_date.strftime("%Y-%m-%d") in known_params:
            return []

    MIN_DATE = df['date'].min()
    AREAS = np.sort(df['key'].unique())
    N_TRAIN = df[df['date'] <= reporting_date]['days'].max() + 1

    VAL_LEN = df['days'].max() - N_TRAIN + 1
    if VAL_LEN < 1:
        return []

    res = []
    for target_name in ['confirmed', 'deaths', 'recovered']:
        df_p = df.pivot(index='key', columns='days', values=target_name).sort_index().values
        model = ZmodelPG().opt(df_p[:, :N_TRAIN], params['loss_fun'],
                                 valid_horizon=params['optim_horizon'], max_trials=params['max_trials'])
        res.append({'date':reporting_date.strftime("%Y-%m-%d"),
                     'target':target_name, 'params':model.params})
    return res

def fit_pg(in_file=pg_params_file, out_file=pg_params_file,
          params = {'loss_fun':loss_rmsle, 'optim_horizon':21, 'max_trials':300}):
    
    # Read the input data
    df = pd.read_csv(country_file)\
            [['date', 'key', 'confirmed', 'deaths', 'recovered', 'confirmed_new', 'active']]
    df['date'] = df['date'].apply(lambda x: (datetime.datetime.strptime(x, '%m/%d/%y')))
    df['days'] = (df['date'].dt.date - df['date'].dt.date.min()).dt.days
    #population = pd.read_csv(bt_file_population)

    # Check if a file with fitted parameters exists and load it
    try:
        with open(in_file, "r") as read_file:
            pg_params = json.load(read_file)
    except:
        pg_params = {}
        
    reporting_dates = [df['date'].min() + timedelta(days=30+i) for i in range(df['days'].max())]
    #reporting_dates = [df['date'].min() + timedelta(days=30+i) for i in range(2)]
    
    # Run backtesting
    res = Parallel(n_jobs=N_JOBS)(delayed(single_bt_fit_pg)\
        (df, reporting_date, params, pg_params)
        for reporting_date in reporting_dates
        )

    # Extract parameters
    res_params = [item for sublist in res for item in sublist if item is not None]

    # Restructure the list of parameters
    for z in res_params:
        if z['date'] not in pg_params:
            pg_params[z['date']] = {}
        pg_params[z['date']][z['target']] = z['params']

    # Save the results
    if out_file is not None:
        with open(out_file, 'w') as fout:
            json.dump(pg_params, fout)

    return pg_params



def single_back_eval_seird(df, reporting_date, params):

    preds_all = []
    for key, df_real in df.groupby('key'):

        df_short = df_real[df_real['date'] <= reporting_date]

        VAL_LEN = df['days'].max() - df_short['days'].max()
        if (VAL_LEN < 1) or (reporting_date.strftime("%Y-%m-%d") not in params):
            return None

        par = list(params[reporting_date.strftime("%Y-%m-%d")][key].values())
        preds = ZmodelSEIRD().predict(df_short, par, VAL_LEN)
        preds = preds[preds['t'] >= len(df_real)-VAL_LEN]

        preds['key'] = key
        preds['prediction_horizon'] = np.arange(len(preds)) + 1
        preds['days'] = preds['prediction_horizon'] + df_short['days'].max()
        preds['reporting_date'] = reporting_date
        preds['date'] = preds['reporting_date'] + pd.to_timedelta(preds['prediction_horizon'], unit='days')
        preds['date_eow'] = preds['date'] - pd.to_timedelta(preds['date'].dt.weekday, 
                                                            unit='days') + datetime.timedelta(days=6)

        real = preds[['key', 'date']].merge(df[['key', 'date', 'confirmed', 'recovered', 'deaths']],
                                            how='left', on=['key', 'date'])
        real = real.melt(id_vars = ['key', 'date'], value_vars = ['confirmed', 'recovered', 'deaths'],
                         value_name='realized', var_name='target')

        preds = preds.melt(id_vars = ['key', 'reporting_date', 'prediction_horizon', 'days', 'date', 'date_eow'],
                   value_vars = ['confirmed', 'recovered', 'deaths'], value_name='predicted', var_name='target')

        preds = preds.merge(real, how='left', on=['key', 'date', 'target'])
        preds['model'] = 'SEIRD'

        preds_all.append(preds)

    return pd.concat(preds_all)

def single_back_eval_pg(df, reporting_date, params):

    preds = []
    for key, df_real in df.groupby('key'):

        df_short = df_real[df_real['date'] <= reporting_date]

        VAL_LEN = df['days'].max() - df_short['days'].max()
        if (VAL_LEN < 1) or (reporting_date.strftime("%Y-%m-%d") not in params):
            return None

        for tgt in ['confirmed', 'deaths', 'recovered']:
            model = ZmodelPG()
            model.params = params[reporting_date.strftime("%Y-%m-%d")][tgt]

            preds_delta = model.predict(df_short[tgt].values.reshape(1,-1), VAL_LEN)
            preds_delta = pd.DataFrame({
                            'key':key,
                            'reporting_date':reporting_date,
                            'model':'PG',
                            'date':df_real['date'].values,
                            'days':df_real['days'].values,
                            'target':tgt,
                            'realized':df_real[tgt].values,
                            'predicted':preds_delta.reshape(-1)})
            preds_delta = preds_delta[preds_delta['date'] > preds_delta['reporting_date']]
            preds_delta['prediction_horizon'] = np.arange(1, VAL_LEN+1)
            preds_delta['date_eow'] = preds_delta['date'] - \
                    pd.to_timedelta(preds_delta['date'].dt.weekday, unit='days') + datetime.timedelta(days=6)
            preds.append(preds_delta)

    return pd.concat(preds)


def single_back_eval_baseline(df, reporting_date, params):

    preds = []
    for key, df_real in df.groupby('key'):

        df_short = df_real[df_real['date'] <= reporting_date]

        VAL_LEN = df['days'].max() - df_short['days'].max()
        if (VAL_LEN < 1):
            return None

        for tgt in ['confirmed', 'deaths', 'recovered']:
            model = ZmodelBaseline()

            preds_delta = model.predict(df_short[tgt].values.reshape(1,-1), VAL_LEN)
            preds_delta = pd.DataFrame({
                            'key':key,
                            'reporting_date':reporting_date,
                            'model':'Baseline',
                            'date':df_real['date'].values,
                            'days':df_real['days'].values,
                            'target':tgt,
                            'realized':df_real[tgt].values,
                            'predicted':preds_delta.reshape(-1)})
            preds_delta = preds_delta[preds_delta['date'] > preds_delta['reporting_date']]
            preds_delta['prediction_horizon'] = np.arange(1, VAL_LEN+1)
            preds_delta['date_eow'] = preds_delta['date'] - \
                    pd.to_timedelta(preds_delta['date'].dt.weekday, unit='days') + datetime.timedelta(days=6)
            preds.append(preds_delta)

    return pd.concat(preds)


def single_back_eval(df, dt, params, model):
    if model == 'SEIRD':
        return single_back_eval_seird(df, dt, params)
    elif model == 'PG':
        return single_back_eval_pg(df, dt, params)
    elif model == 'Baseline':
        return single_back_eval_baseline(df, dt, params)
    else:
        return None


def make_bt_plots():

    df = pd.read_csv(country_file)\
            [['date', 'key', 'confirmed', 'deaths', 'recovered', 'confirmed_new', 'active']]
    df['date'] = df['date'].apply(lambda x: (datetime.datetime.strptime(x, '%m/%d/%y')))
    df['days'] = (df['date'].dt.date - df['date'].dt.date.min()).dt.days

    population = pd.read_csv(bt_file_population)
    
    with open(seird_params_file, "r") as read_file:
        seird_params = json.load(read_file)
    with open(pg_params_file, "r") as read_file:
        pg_params = json.load(read_file)
    

    res = Parallel(n_jobs=N_JOBS, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing")\
        (delayed(single_back_eval)\
            (df, reporting_date, params, model)
                for reporting_date in [df['date'].min() + timedelta(days=30+i) for i in range(df['days'].max())] \
                for model, params in [('SEIRD', seird_params), ('PG', pg_params), ('Baseline', None)]
        )

    res_df = pd.concat([x for x in res if x is not None])
    
    
    
    def gen_agg_losses(df, loss_f_list, horizons=[7,14,21,28], min_cases=[0,100,1000,5000], date_col='reporting_date'):
        res = []
        for mc in min_cases:
            for hor in horizons:
                for loss_f in loss_f_list:
                    df2 = df[(df['prediction_horizon'] == hor) & (df['realized'] >= mc)]
                    l = df2.groupby(['model', date_col, 'target']).\
                        apply(lambda z: loss_f(z['realized'].values, z['predicted'].values)).rename("loss").reset_index()
                    l['loss_func'] = loss_f.__doc__
                    l['prediction_horizon'] = hor
                    l['min_cases'] = mc
                    res.append(l)
        return pd.concat(res)

    agg_losses = gen_agg_losses(res_df, [loss_rmsle, loss_mape], date_col='reporting_date')
    #agg_losses = gen_agg_losses(res_df[res_df['key'] != 'Tajikistan'], [loss_rmsle, loss_mape], date_col='reporting_date')
    global_confirmed = df.groupby('date')['confirmed_new'].sum()
    global_confirmed = global_confirmed[global_confirmed.index >= agg_losses['reporting_date'].min()]
    agg_losses.to_csv(bt_file_by_date, index=False)



    def gen_agg_losses2(df, loss_f_list, horizons=[7,14,21,28], date_col='reporting_date'):
        df0 = df.merge(df[(df['prediction_horizon']==1) & (df['model'] == 'Baseline')]\
                 [['key', 'target', 'date', 'realized']].\
                 rename({'realized':'forecast_realized', 'date':'reporting_date'}, axis=1), \
                 how='left', on=['key', 'target', 'reporting_date'])
        df0 = df0[~df0['forecast_realized'].isnull()]
        res = []
        for hor in horizons:
            for loss_f in loss_f_list:
                df2 = df0[df0['prediction_horizon'] == hor]
                l = df2.groupby([df2['model'], 
                        (2**np.floor(np.log2(df2['forecast_realized']))).astype(int).rename('bucket'),
                        df2['target']]).\
                        apply(lambda z: pd.Series({'loss': loss_f(z['realized'].values, z['predicted'].values), 'n':len(z)})).\
                        reset_index()
                l['loss_func'] = loss_f.__doc__
                l['prediction_horizon'] = hor
                res.append(l)
        return pd.concat(res)

    agg_losses2 = gen_agg_losses2(res_df, [loss_rmsle, loss_mape], date_col='reporting_date')
    agg_losses2.to_csv(bt_file_by_num_cases, index=False)
    
    
    
    def gen_area_losses(df, loss_f_list, horizons=[7,14,21,28]):
        res = []
        for hor in horizons:
            for loss_f in loss_f_list:
                #temp = df[(df['model'] != 'Baseline') & (df['target'] == 'confirmed')]
                temp = df[(df['target'] == 'confirmed')]
                temp = temp[temp['prediction_horizon'] == hor].groupby(['key', 'model', 'date_eow', 'target'])
                l = temp.apply(lambda z: loss_f(z['realized'].values, z['predicted'].values)).rename("loss").reset_index()
                l['real'] = temp['realized'].max().values
                l['loss_func'] = loss_f.__doc__
                l['prediction_horizon'] = hor
                res.append(l)
        return pd.concat(res)
    area_losses = gen_area_losses(res_df[res_df['date_eow'].isin(np.sort(res_df['date_eow'].unique())[-10:])],
                                  [loss_rmsle])
    area_losses['current_cases'] = area_losses['key'].map(df.groupby('key')['confirmed'].max())
    area_losses = area_losses.merge(population, how='left', left_on='key', right_on='key')
    area_losses.to_csv(bt_file_by_area, index=False)
    
    
    
    area_real_vs_pred = res_df[(res_df['target'] == 'confirmed') & (res_df['prediction_horizon'].isin([7,14,21,28])) \
                    & (res_df['date'] == res_df['date_eow'])]\
                    [['key', 'predicted', 'date', 'realized', 'model', 'prediction_horizon']]

    area_real_vs_pred = area_real_vs_pred[area_real_vs_pred['date'].isin(np.sort(area_real_vs_pred['date'].unique())[-10:])]
    area_real_vs_pred = area_real_vs_pred.merge(population, how='left', left_on='key', right_on='key')
    area_real_vs_pred.to_csv(bt_file_real_vs_pred, index=False)
    
    return


if __name__ == "__main__":
    start_time = time.time()
    #fit_seird()
    #fit_pg()
    make_bt_plots()
    print(f"Elapsed {time.time() - start_time : 8.5f} s")


