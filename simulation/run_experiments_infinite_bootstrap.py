import numpy as np
import os, time, sys

sys.path.append("../")

os.environ["MKL_NUM_THREADS"] = "1"

from simulation.simulate_data import (
    generate_model_parameters,
    simulate_dataset_infinite,
)
from cores.estimate_mediation_effect_infinite import (
    estimate_mediation_j_effect_infinite,
    bootstrap_infinite,
)
from cores.estimate_mediation_effect_infinite_baseline import (
    estimate_mediation_j_effect_infinite_conditional_independence,
    estimate_mediation_j_effect_infinite_ignore_time_dependence,
    estimate_mediation_j_effect_infinite_ignore_time_dependence_wrapper,
)
from cores.estimate_model_parameters import estimate_model_parameters
from cores.calculate_true_eta import calculate_true_etaj_infinite
from p_tqdm import p_map
from utils import *

CORES = 5


def mc_true_eta_j(j, params, nsim_mc, n, T):
    def make_one_run_true(j, n, params, seed):
        np.random.seed(seed)
        params["n"] = n
        params["T"] = T
        etaj_true = calculate_true_etaj_infinite(j=j, params=params)
        return etaj_true

    outs = p_map(
        make_one_run_true,
        [j] * nsim_mc,
        [n] * nsim_mc,
        [params] * nsim_mc,
        range(nsim_mc),
        desc="true j={}".format(j),
        num_cpus=CORES,
    )
    etaj_true = np.mean(np.array([x for x in outs]), axis=0)
    return etaj_true


# ------ experiment 6: main result in paper ----- #


def make_one_run_exp6(j, actions, mediators, rewards, w_threshold, B):
    n, T = actions.shape
    params_est = estimate_model_parameters(
        actions=actions,
        mediators=mediators,
        rewards=rewards,
        estimate_W0=True,
        estimate_W0_lambda1=0.00,
        estimate_W0_threshold=w_threshold,
    )
    etaj_est_ours = estimate_mediation_j_effect_infinite(
        j=j, actions=actions, mediators=mediators, rewards=rewards, params=params_est
    )
    q0_ours, q1_ours, bstd_ours = bootstrap_infinite(
        estimator=estimate_mediation_j_effect_infinite,
        j=j,
        actions=actions,
        mediators=mediators,
        rewards=rewards,
        cores=1,
        B=B,
        w_threshold=w_threshold,
        W0=params_est["W0"],
        flag_estimate_W0=False,
    )
    etaj_est_conditional_independence = (
        estimate_mediation_j_effect_infinite_conditional_independence(
            j=j,
            actions=actions,
            mediators=mediators,
            rewards=rewards,
            params=params_est,
        )
    )
    (
        q0_conditional_independence,
        q1_conditional_independence,
        bstd_conditional_independence,
    ) = bootstrap_infinite(
        estimator=estimate_mediation_j_effect_infinite_conditional_independence,
        j=j,
        actions=actions,
        mediators=mediators,
        rewards=rewards,
        cores=1,
        B=B,
        w_threshold=w_threshold,
        flag_estimate_W0=False,
    )
    etaj_est_time_independence, paj = (
        estimate_mediation_j_effect_infinite_ignore_time_dependence(
            j=j, actions=actions, mediators=mediators, rewards=rewards, paj=None
        )
    )
    q0_time_independence, q1_time_independence, bstd_time_independence = (
        bootstrap_infinite(
            estimator=estimate_mediation_j_effect_infinite_ignore_time_dependence_wrapper,
            j=j,
            actions=actions,
            mediators=mediators,
            rewards=rewards,
            cores=1,
            B=B,
            w_threshold=w_threshold,
            paj=paj,
            flag_estimate_W0=False,
        )
    )

    return (
        np.array(
            [
                etaj_est_ours,
                etaj_est_conditional_independence,
                etaj_est_time_independence,
            ]
        ),
        np.array([q0_ours, q0_conditional_independence, q0_time_independence]),
        np.array([q1_ours, q1_conditional_independence, q1_time_independence]),
        np.array([bstd_ours, bstd_conditional_independence, bstd_time_independence]),
    )


def run_exp6():
    note = ""
    seed = 1
    nsim = 500
    # nsim = 2
    d = 3
    Js = list(range(d))
    # Js = [1]
    p_M = 0.9
    p_A = 0.5
    w_threshold = 0.1
    B = 100
    params = generate_model_parameters(
        n=np.nan, T=np.nan, d=d, p_M=p_M, p_A=p_A, seed=seed
    )
    print(params["W0"])

    suffix = datetime_suffix() + note
    res0 = expand_grid({"n": [20, 50, 100], "j": Js, "T": [100, 250, 500]})
    # res0 = expand_grid({"n": [20], "j": Js, "T": [100]})
    res = pd.DataFrame({})
    for idx, row in res0.iterrows():
        t0 = time.time()
        params["n"], j, params["T"] = int(row["n"]), int(row["j"]), int(row["T"])
        true_etaj = mc_true_eta_j(j=j, params=params, nsim_mc=100, n=10000, T=1000)
        mediators, actions, rewards = simulate_dataset_infinite(
            params=params, nsim=nsim
        )
        outs = p_map(
            make_one_run_exp6,
            [j] * nsim,
            [actions[i] for i in range(nsim)],
            [mediators[i] for i in range(nsim)],
            [rewards[i] for i in range(nsim)],
            [w_threshold] * nsim,
            [B] * nsim,
            num_cpus=CORES,
            desc="n:{},j:{},T:{}".format(row["n"], j, row["T"]),
        )

        labels = ["ours", "conditional_independence", "time_independence"]
        t1 = time.time()
        for i in range(len(labels)):
            etajs_est = np.array([x[0][i] for x in outs])
            q0s = np.array([x[1][i] for x in outs])
            q1s = np.array([x[2][i] for x in outs])
            bstds = np.array([x[3][i] for x in outs])
            bias = etajs_est - true_etaj
            # ese = np.std(etajs_est)
            # rmse = np.sqrt(np.square(np.subtract(etajs_est, true_etaj)).mean(axis=0))
            df = pd.DataFrame(
                {
                    "label": [labels[i]] * nsim,
                    "n": [row["n"]] * nsim,
                    "d": [d] * nsim,
                    "T": [row["T"]] * nsim,
                    "j": [j] * nsim,
                    "true": [true_etaj] * nsim,
                    "est": etajs_est,
                    "bias": bias,
                    "q0": q0s,
                    "q1": q1s,
                    "bstd": bstds,
                    "covered": np.logical_and(
                        np.less_equal(q0s, true_etaj),
                        np.greater_equal(q1s, true_etaj),
                    ),
                    "time": [t1 - t0] * nsim,
                }
            )
            # df.to_csv(
            #     "./outs/tmp_infinite_bs_exp6_d{}_seed{}_{}.csv".format(d, seed, suffix),
            #     index=False,
            #     header=False,
            #     mode="a",
            # )
            res = pd.concat([res, df])

    res.to_csv(
        "./outs/infinite_bs_exp6_d{}_seed{}_{}.csv".format(d, seed, suffix), index=False
    )
    # print(res)


if __name__ == "__main__":
    run_exp6()
