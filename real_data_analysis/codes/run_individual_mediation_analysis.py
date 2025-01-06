import sys, os, subprocess

sys.path.extend(["../../"])

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, splrep, make_smoothing_spline

from datasets import IHS2018Dataset, IHS2018DatasetMI, IHS2020DatasetMI
from cores.estimate_mediation_effect_finite import (
    estimate_mediation_j_effect_finite_multiple_stages_recursion,
)

from estimate_mediation_effect_finite_plus import (
    bootstrap_finite,
    estimate_mediation_j_effect_finite_multiple_stages_recursion_wrapper,
)

import matplotlib.pyplot as plt
from utils import *
import networkx as nx

mylogger = create_logger("mediator_analysis", logging_level="info")
prefix = set_home_folder_prefix(
    {
        "default": "/home/jitwang/dynamic_mediation_project/python_version/real_data_analysis/",
    }
)


def smooth_x_y(x, y):
    x, y = np.array(x), np.array(y)
    tck = splrep(x, y, t=[7, 14, 21], k=3)
    spline = BSpline(tck[0], tck[1], tck[2], extrapolate=False)
    # spline = make_smoothing_spline(x, y, lam=50)
    x_smooth = np.linspace(x.min(), x.max(), 500)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth


def run_ihs2018(imp_iter, smooth=True, msg_type="activity"):
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
        folder = os.path.join(prefix, "outs/ihs2018_{}_{}".format(msg_type, specialty))
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
        mediators, actions, rewards = dataset.load_data(index=imp_iter)
        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=[9, 4],
            gridspec_kw={"width_ratios": [3, 3], "wspace": 0.4},
        )
        fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=[5, 4])
        res = pd.DataFrame()
        for j in range(dataset.num_mediators):
            mylogger.info(
                "running mediator analysis - {} - {} - N({}) - T({})".format(
                    specialty, dataset.mediator_names[j], dataset.N, dataset.T
                )
            )
            etajs, etajs_immediate, etajs_lagged, _ = (
                estimate_mediation_j_effect_finite_multiple_stages_recursion(
                    j=j,
                    actions=actions,
                    mediators=mediators,
                    rewards=rewards,
                    w_threshold=0.1,
                    singleW=True,
                )
            )
            etajs_sum = etajs_immediate + etajs_lagged
            res0 = pd.DataFrame(
                data={
                    "t": range(len(etajs)),
                    "IME": etajs,
                    "IIME": etajs_immediate,
                    "DIME": etajs_lagged,
                    "sum": etajs_sum,
                    "mediator": dataset.mediator_names[j],
                }
            )
            res = pd.concat([res, res0])

            x, y = (
                smooth_x_y(range(len(etajs_immediate)), etajs_immediate)
                if smooth
                else (range(len(etajs_immediate)), etajs_immediate)
            )
            axs[0].plot(
                x,
                y,
                label="{}".format(names_map[dataset.mediator_names[j]]),
                color=cmap(j),
            )

            x, y = (
                smooth_x_y(range(len(etajs_lagged)), etajs_lagged)
                if smooth
                else (range(len(etajs_lagged)), etajs_lagged)
            )
            axs[1].plot(
                x,
                y,
                label="{}".format(names_map[dataset.mediator_names[j]]),
                color=cmap(j),
            )
            ax2.plot(
                x,
                y,
                label="{}".format(names_map[dataset.mediator_names[j]]),
                color=cmap(j),
            )
        res.to_csv(
            os.path.join(folder, "res_finite_{}_{}.csv".format(specialty, imp_iter))
        )
        axs[0].set_ylabel("IIME")

        axs[0].autoscale()
        axs[0].axhline(y=0.0, linestyle="--", color="black")
        # axs[1].legend()
        axs[1].set_ylabel("DIME")
        axs[1].autoscale()

        axs[1].axhline(y=0.0, linestyle="--", color="black")
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", ncols=4, bbox_to_anchor=[0.5, -0.1]
        )
        plt.tight_layout()
        figure_name = (
            "ihs2018_smooth_{}.png".format(imp_iter)
            if smooth
            else "ihs2018_raw_{}.png".format(imp_iter)
        )
        fig.savefig(os.path.join(folder, figure_name), bbox_inches="tight")

        ax2.set_ylabel(r"$\Delta_j^{(t)}$")
        ax2.autoscale()
        ax2.axhline(y=0.0, linestyle="--", color="black")
        ax2.legend()
        ax2.set_xlabel("Week")
        fig2.savefig(
            os.path.join(folder, "ihs2018_smooth_etasum.png"), bbox_inches="tight"
        )
    p = subprocess.Popen(
        ["Rscript", prefix + "codes/plot_IIME_DIME.R", folder], stdout=subprocess.PIPE
    )
    p.wait()


def run_ihs2018MI(smooth=True, msg_type="all"):
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
            prefix, "outs/ihs2018MI_{}_{}".format(msg_type, specialty)
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
        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=[9, 4],
            gridspec_kw={"width_ratios": [3, 3], "wspace": 0.4},
        )
        fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=[5, 4])
        res = pd.DataFrame()
        W_est_pool = np.zeros([4, 4])
        # for imp_iter in range(dataset.num_imputations):
        for imp_iter in range(1):
            imp_iter = 10
            mediators, actions, rewards = dataset.load_data(index=imp_iter)
            for j in range(dataset.num_mediators):
                mylogger.info(
                    "MI:{} - running mediator analysis - {} - {} - N({}) - T({})".format(
                        imp_iter,
                        specialty,
                        dataset.mediator_names[j],
                        dataset.N,
                        dataset.T,
                    )
                )
                etajs, etajs_immediate, etajs_lagged, W_est = (
                    estimate_mediation_j_effect_finite_multiple_stages_recursion(
                        j=j,
                        actions=actions,
                        mediators=mediators,
                        rewards=rewards,
                        w_threshold=0.1,
                        singleW=True,
                    )
                )
                # TODO modify the function to return IIME DIME
                (
                    etajs_q0,
                    etajs_q1,
                    etajs_std,
                    etajs_immediate_q0,
                    etajs_immediate_q1,
                    etajs_immediate_std,
                    etajs_lagged_q0,
                    etajs_lagged_q1,
                    etajs_lagged_std,
                ) = bootstrap_finite(
                    estimator=estimate_mediation_j_effect_finite_multiple_stages_recursion_wrapper,
                    j=j,
                    actions=actions,
                    mediators=mediators,
                    rewards=rewards,
                    w_threshold=0.1,
                    singleW=True,
                    B=100,
                    cores=5,
                )
                W_est_pool = W_est_pool + W_est
                etajs_sum = etajs_immediate + etajs_lagged
                res0 = pd.DataFrame(
                    data={
                        "t": range(len(etajs)),
                        "IME": etajs,
                        "IME_q0": etajs_q0,
                        "IME_q1": etajs_q1,
                        "IME_std": etajs_std,
                        "IIME": etajs_immediate,
                        "IIME_q0": etajs_immediate_q0,
                        "IIME_q1": etajs_immediate_q1,
                        "IIME_std": etajs_immediate_std,
                        "DIME": etajs_lagged,
                        "DIME_q0": etajs_lagged_q0,
                        "DIME_q1": etajs_lagged_q1,
                        "DIME_std": etajs_lagged_std,
                        "sum": etajs_sum,
                        "mediator": dataset.mediator_names[j],
                        "imp_iter": imp_iter,
                    }
                )
                res = pd.concat([res, res0])
    res.to_csv(os.path.join(folder, "res_finite_{}.csv".format(specialty)))
    p = subprocess.Popen(
        ["Rscript", prefix + "codes/plot_IIME_DIME_MI.R", folder],
        stdout=subprocess.PIPE,
    )
    p.wait()

    W_est_pool = W_est_pool / dataset.num_imputations
    for i in range(dataset.num_mediators):
        W_est_pool[i, i] = 0
    G = nx.from_numpy_array(W_est_pool, create_using=nx.DiGraph)
    mapping = {0: "step", 1: "sleep", 2: "RHR", 3: "HRV"}
    G = nx.relabel_nodes(G, mapping)
    colors = ["r" if G[u][v]["weight"] > 0 else "b" for u, v in G.edges()]
    pos = nx.circular_layout(G)
    nx.draw(G, with_labels=True, node_size=1000, edge_color=colors, pos=pos)
    plt.savefig(os.path.join(folder, "dag.png"))
    print(W_est_pool)


if __name__ == "__main__":
    for msg_type in ["all"]:
        # run_ihs2018(imp_iter=1, msg_type=msg_type, smooth=False)
        run_ihs2018MI(msg_type=msg_type, smooth=True)
