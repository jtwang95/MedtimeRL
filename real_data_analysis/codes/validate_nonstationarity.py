import sys, os, subprocess

sys.path.extend(
    [
        "/home/jitwang/Dropbox (University of Michigan)/dynamic_mediation_project/dynamic_mediation_project/python_version/"
    ]
)
import numpy as np
import matplotlib.pyplot as plt
from cores.estimate_mediation_effect_finite import (
    estimate_parent,
    estimate_stagewise_model_parameters,
)
import networkx as nx
from datasets import IHS2018DatasetMI
from utils import *
import pandas as pd
from tqdm import trange

prefix = set_home_folder_prefix(
    {
        "x300": "/home/jitwang/Dropbox (University of Michigan)/dynamic_mediation_project/dynamic_mediation_project/python_version/real_data_analysis/",
        "default": "/home/jitwang/dynamic_mediation_project/python_version/real_data_analysis/",
    }
)


def run_MI_stagewise_param(j, actions, mediators, rewards):
    n, T, d = mediators.shape
    d = pd.DataFrame()
    for t in range(1, T):
        params_est = estimate_stagewise_model_parameters(
            action_t=actions[:, t],
            mediator_t=mediators[:, t, :],
            reward_t=rewards[:, t],
            mediator_t1=mediators[:, t - 1, :],
            reward_t1=rewards[:, t - 1],
            estimate_W0=False,
            estimate_W0_lambda1=0.00,
        )
        alpha1, beta1, Gamma1, zeta1, alpha2, beta2, zeta2, gamma2, kappa = (
            params_est["alpha1"],
            params_est["beta1"],
            params_est["Gamma1"],
            params_est["zeta1"],
            params_est["alpha2"],
            params_est["beta2"],
            params_est["zeta2"],
            params_est["gamma2"],
            params_est["kappa"],
        )
        for key, value in params_est.items():
            value = value.flatten()
            for i in range(len(value)):
                tmp = pd.DataFrame(
                    {
                        "key": [key],
                        "value": [value[i]],
                        "j": [j],
                        "t": [t],
                        "idx": [i],
                    }
                )
                d = pd.concat([d, tmp], ignore_index=True)

    return d


def run_ihs2018MI(msg_type="all"):
    specialties = ["all"]
    msg_type = "all"
    cmap = plt.get_cmap("tab10")
    names_map = {
        "MOOD": "Mood",
        "STEP_COUNT": "Step",
        "SLEEP_COUNT": "Sleep",
        "resting_hr": "RHR",
        "rmssd": "HRV",
    }
    for specialty in specialties:
        folder = os.path.join(
            prefix, "outs/nonstationarity_{}_{}".format(msg_type, specialty)
        )
        try:
            os.mkdir(folder)
        except:
            pass
        dataset = IHS2018DatasetMI(
            specialty=specialty,
            msg_type=msg_type,
            reward_name="MOOD",
            mediator_names=["STEP_COUNT", "SLEEP_COUNT", "resting_hr", "rmssd"],
        )
        res = pd.DataFrame()
        for imp_iter in trange(dataset.num_imputations):
            mediators, actions, rewards = dataset.load_data(index=imp_iter)
            for j in range(dataset.num_mediators):
                res0 = run_MI_stagewise_param(j, actions, mediators, rewards)

                res = pd.concat([res, res0], ignore_index=True)
    res.to_csv(os.path.join(folder, "param_est_MI_finite_{}.csv".format(specialty)))


if __name__ == "__main__":
    run_ihs2018MI()
