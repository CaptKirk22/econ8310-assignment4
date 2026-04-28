import pandas as pd

data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/cookie_cats.csv")

import pymc as pm

data["treatment"] = (data['version'] == 'gate_40').astype(int)

data

retention_obs = data["retention_7"].values #change to retention_1 for other AB test


with pm.Model() as model:
  lambda_30 = pm.Beta("p_30", alpha =2, beta =2)
  lambda_40 = pm.Beta("p_40", alpha =2, beta =2)

  lambda_ = pm.math.switch(data["treatment"], lambda_40, lambda_30)

  observation = pm.Bernoulli("obs", lambda_, observed = retention_obs)

  delta = pm.Deterministic("delta", lambda_40 - lambda_30)

  trace = pm.sample(10000, tune=5000,
        target_accept=0.9, chains = 2, return_inferencedata=True)

with model:
  pm.plot_forest(trace, kind='ridgeplot',var_names=['p_30', 'p_40'],combined=True)
  pm.plot_forest(trace, kind='ridgeplot',var_names=['delta'],combined=True)


