import numpy as np
import pandas as pd
import multiprocessing as mp
from hyperopt import hp, space_eval, fmin, tpe, Trials, rand
from hyperopt.pyll.base import scope
from joblib import Parallel, delayed
from collections import Iterable
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from scipy.optimize import differential_evolution
from scipy.integrate import odeint


############### PG MODEL ####################


class ZmodelBase():
    def __init__(self):
        self.space = {}

    def _predict(self, params, X, horizon):
        pass

    def _objective(self, params, X, horizon):
        preds = self._predict(params, X[:,:-horizon], horizon)
        return self.loss_fun(X[:, -horizon:], preds[:, -horizon:])
    
    def _objective_set(self, params, X, horizon, step=7, min_size=30):
        if X.shape[1] <= min_size + horizon:
            return self._objective(params, X, horizon)
        
        return np.mean([self._objective(params, X[:, :X.shape[1] - i], horizon) \
                        for i in np.arange(0, X.shape[1] - min_size, step)])

    def opt(self, X, loss_fun, valid_horizon=14, rstate=42, max_trials=30, overrides={}):
        self.loss_fun = loss_fun
        for key, value in overrides.items():
            self.space[key] = value
        trials = Trials()
        rstate = np.random.RandomState(rstate)

        best = fmin(
                    #lambda p: self._objective(p, X, valid_horizon),
                    lambda p: self._objective_set(p, X, valid_horizon),
                    self.space,
                    algo=tpe.suggest,
                    max_evals=max_trials,
                    trials=trials,
                    rstate=rstate,
                    show_progressbar=False,
                    verbose=0)
        self.params = space_eval(self.space, best)
        self.opt_loss = self._objective(self.params, X, valid_horizon)
        return self

    def predict(self, X, test_horizon=50):
        return self._predict(self.params, X, test_horizon)


class ZmodelPG(ZmodelBase):
    "PG model"
    def __init__(self):
        super(ZmodelPG, self).__init__()
        self.space = {
            'min_cases_for_growth_rate': hp.choice('min_cases_for_growth_rate', [0,10,50]),
            'last_n_days': hp.choice('last_n_days', [7,14,21]),
            'growth_rate_max': hp.quniform('growth_rate_max', 0.0, 0.5, 0.1),
            'growth_rate_default': hp.quniform('growth_rate_default', 0.0, 0.5, 0.01),
            'growth_rate_decay': hp.quniform('growth_rate_decay', -1.0, 0.0, 0.01),
            'growth_rate_decay_acceleration': hp.quniform('growth_rate_decay_acceleration', 0.0, 1.0, 0.01),
        }

    def _predict(self, params, X0, horizon):
        X = np.log(1+X0)

        gr_base = []
        gr_base_factor = []

        for i in range(X.shape[0]):
            temp = X[i, :]
            threshold = np.log(1 + params['min_cases_for_growth_rate'])
            num_days = params['last_n_days']
            if (temp > threshold).sum() > num_days:
                d = np.diff(temp[temp > threshold])[-num_days:]
                w = np.arange(len(d)) + 1
                w = w ** 5
                w = w / np.sum(w)
                gr_base.append(np.clip(np.average(d, weights=w), 0, params['growth_rate_max']))
            else:
                gr_base.append(params['growth_rate_default'])

        gr_base = np.array(gr_base)
        preds = X.copy()

        for i in range(horizon):
            delta = np.clip(preds[:, -1], 0, None) + \
                gr_base * np.clip(1 + params['growth_rate_decay'] * 
                                  (1 + params['growth_rate_decay_acceleration']) ** (i), 0, None) ** (np.log1p(i))
            preds = np.hstack((preds, delta.reshape(-1, 1)))
        return np.exp(preds) - 1


class ZmodelBaseline(ZmodelBase):
    "Baseline"
    def __init__(self):
        super(ZmodelBaseline, self).__init__()

    def opt(self, X, loss_fun, valid_horizon=14, rstate=42, max_trials=30, overrides={}):
        self.loss_fun = loss_fun
        self.params = {}
        return self

    def predict(self, X, test_horizon=50):
        preds = X.copy()
        for i in range(test_horizon):
            preds = np.hstack((preds, preds[:, -1].reshape(-1,1) ))
        return preds


    
class ZmodelSEIRD(ZmodelBase):
    "SEIRD dummy"
    def __init__(self):
        super(ZmodelSEIRD, self).__init__()

    def opt(self):
        pass

    def predict(self, train, params, test_horizon=50):
        model = EpidemicModel(EpidemicModelParams())
        model.best_params = params[:9]
        model = model.fit(train, workers=1, skip_opt=True)
        return model.predict(test_horizon, False)




############### Updated SEIRD ####################

class EpidemicModelParams(object):
    #def __init__(self, N=(100, 1000000), beta=(1, 1.5), gamma=(1e-7, 0.5),
    def __init__(self, N=(100, 1000000), beta=(0.5, 3.0), gamma=(1e-7, 0.5),
                 delta=(0.01, 0.5), alpha=(0.01, 0.2), rho=(0.01, 0.4), lockdown=(-1, 100),
                 beta_decay=(0, 1), beta_decay_rate=(0.01, 3), adjust_bounds=True):
        """
            N: total population
            beta : infection rate
            gamma: recovery rate
            delta: incubation period
            alpha: fatality rate
            rho: rate at which people die
            lockdown: day of lockdown (-1 => no lockdown)
            beta_decay: beta decay due to lockdown
            beta_decay_rate: speed of beta decay

            adjust_bounds: whether model is allowed to adjust param bounds
        """
        self.bounds = locals()
        self.bounds.pop('self')
        self.names = []
        self.adjust_bounds = adjust_bounds

        for p in self.bounds:
            self.names.append(p)
            if not isinstance(self.bounds[p], Iterable):
                self.bounds[p] = (self.bounds[p], self.bounds[p])


class EpidemicModel(object):
    def __init__(self, params, confirmed_col='confirmed', active_col='active',
                 recovered_col='recovered', deaths_col='deaths'):
        self.param_bounds = params.bounds
        self.best_params = None
        self.best_loss = None
        self.confirmed_col = confirmed_col
        self.active_col = active_col
        self.recovered_col = recovered_col
        self.deaths_col = deaths_col
        self.adjust_bounds = params.adjust_bounds
        self.fitted = False

    def fit(self, X, workers=1, popsize=15, skip_opt=False, **fit_params):
        self.n = X.shape[0]
        y = X[[self.active_col, self.recovered_col, self.deaths_col]].values
        y[y < 0] = 0
        x = np.arange(self.n)

        non_zero_active = np.argwhere(y.T[0] > 0)
        t0 = 0
        if len(non_zero_active):
            t0 = non_zero_active.min()
            if t0 >= self.n - 1:
                t0 = 0
        y_t0 = y.T[0][t0]

        if skip_opt:
            self.best_params += [t0, y_t0]
            return self

        N_bounds = self.param_bounds['N']
        beta_bounds = self.param_bounds['beta']
        gamma_bounds = self.param_bounds['gamma']
        delta_bounds = self.param_bounds['delta']
        alpha_bounds = self.param_bounds['alpha']
        rho_bounds = self.param_bounds['rho']
        lockdown_bounds = self.param_bounds['lockdown']
        beta_decay_bounds = self.param_bounds['beta_decay']
        beta_decay_rate_bounds = self.param_bounds['beta_decay_rate']
        t_bounds = (t0, t0)
        y_t0_bounds = (y_t0, y_t0)

        if self.adjust_bounds:
            mx_active = max(1, X[self.active_col].max())
            if N_bounds[0] != N_bounds[1]:
                lb = ub = mx_active
                for i in range(28):
                    ub **= 1.0118 # global mean factor
                ub = min(ub, 10000000)
                N_bounds = (int(lb), int(ub))

        p = differential_evolution(
            self._func_loss, workers=workers, popsize=popsize,
            args=(x, y),
            bounds=(
                N_bounds, beta_bounds, gamma_bounds, delta_bounds, alpha_bounds, rho_bounds,
                lockdown_bounds, beta_decay_bounds, beta_decay_rate_bounds,
                t_bounds, y_t0_bounds
            )
        )
        #assert p.success, "optimization failed"
        self.best_loss = p.fun
        self.best_params = p.x
        self.fitted = True
        self.optres = p
        return self

    def predict(self, horizon=1, horizon_only=True):
        x = np.arange(self.n + horizon)
        yhat = self._func(self.best_params, x)

        ret = pd.DataFrame(x, columns=['t'])
        ret[self.confirmed_col] = np.sum(yhat, axis=1).astype(int)
        ret[self.active_col] = yhat.T[0].astype(int)
        ret[self.recovered_col] = yhat.T[1].astype(int)
        ret[self.deaths_col] = yhat.T[2].astype(int)
        #ret[self.confirmed_col] = np.sum(yhat, axis=1)
        #ret[self.active_col] = yhat.T[0]
        #ret[self.recovered_col] = yhat.T[1]
        #ret[self.deaths_col] = yhat.T[2]

        if horizon_only:
            ret = ret.iloc[-horizon:].reset_index(drop=True)
        return ret

    def best_params_dict(self):
        if self.best_params is None:
            return {}
        param_names = ['N', 'beta', 'gamma', 'delta', 'alpha', 'rho', 'lockdown', 'beta_decay', 'beta_decay_rate']
        param_values = self.best_params[:len(param_names)]
        res = dict(zip(param_names, param_values))
        res['loss'] = self.best_loss
        return res

    @staticmethod
    def _deriv(y, t, N, beta, gamma, delta, alpha, rho, lockdown, beta_decay, beta_decay_rate):

        if lockdown >= 0:
            beta_min = beta * (1 - beta_decay)
            beta = (beta - beta_min) / (1 + np.exp(-beta_decay_rate * (-t + lockdown))) + beta_min

        S, E, I, R, D = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - delta * E
        dIdt = delta * E - (1 - alpha) * gamma * I - alpha * rho * I
        dRdt = (1 - alpha) * gamma * I
        dDdt = alpha * rho * I
        return dSdt, dEdt, dIdt, dRdt, dDdt

    @staticmethod
    def _func(params, x):
        N, beta, gamma, delta, alpha, rho, lockdown, beta_decay, beta_decay_rate, t0, y_t0 = params
        t0 = int(t0)
        preds = np.zeros((x.shape[0], 3))

        I0, R0, E0, D0 = y_t0, 0, 0, 0  # real initial conditions, y[t0] > 0
        S0 = N - I0 - R0 - E0 - D0
        y0 = S0, E0, I0, R0, D0
        t = np.linspace(0, x.max() - t0 - 1, x.max() - t0)
        ret = odeint(EpidemicModel._deriv, y0, t, args=(N, beta, gamma, delta, alpha, rho,
                                                        lockdown, beta_decay, beta_decay_rate))
        S, E, I, R, D = ret.T

        preds[-I.shape[0]:, 0] = I[-x.shape[0]:].astype(int)
        preds[-R.shape[0]:, 1] = R[-x.shape[0]:].astype(int)
        preds[-D.shape[0]:, 2] = D[-x.shape[0]:].astype(int)
        preds[preds < 0] = 0
        #preds[-I.shape[0]:, 0] = I[-x.shape[0]:]
        #preds[-R.shape[0]:, 1] = R[-x.shape[0]:]
        #preds[-D.shape[0]:, 2] = D[-x.shape[0]:]

        return preds

    @staticmethod
    def _func_loss(params, x, y):
        yhat = EpidemicModel._func(params, x)
        loss = []
        for i in range(yhat.shape[1]):
            loss.append(mean_squared_error(y.T[i], yhat.T[i]))
            #loss.append(mean_squared_log_error(y.T[i], yhat.T[i]))
        loss = np.average(list(map(np.sqrt, loss)), weights=[0.33, 0.33, 0.33])
        return loss

