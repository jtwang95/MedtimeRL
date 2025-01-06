import os, time, sys

sys.path.append("../")

os.environ["MKL_NUM_THREADS"] = "1"

from simulation.simulate_data import generate_model_parameters, simulate_dataset_finite
from cores.estimate_mediation_effect_finite import (
    estimate_mediation_j_effect_finite_multiple_stages_recursion_wrapper,
    estimate_mediation_j_effect_finite_multiple_stages_recursion,
    bootstrap_finite,
)
from cores.estimate_mediation_effect_finite_baseline import (
    estimate_mediation_j_effect_finite_multiple_stages_ignore,
    estimate_mediation_j_effect_finite_multiple_stages_independence,
    estimate_mediation_j_effect_finite_multiple_stages_ignore_wrapper,
)

from cores.calculate_true_eta import calculate_true_etaj_finite
from p_tqdm import p_map
from utils import *

CORES = 2


def mc_true_eta_j(j, params_list, nsim_mc, n, T):
    def make_one_run_true(j, n, params_list, seed):
        np.random.seed(seed)
        etaj_true = calculate_true_etaj_finite(j=j, n=n, T=T, params_list=params_list)
        return etaj_true

    outs = p_map(
        make_one_run_true,
        [j] * nsim_mc,
        [n] * nsim_mc,
        [params_list] * nsim_mc,
        range(nsim_mc),
        desc="true j={}".format(j),
        num_cpus=CORES,
    )
    etaj_true = np.mean(np.array([x for x in outs]), axis=0)
    return etaj_true


# ------ experiment 6: bootstrap for three methods ----- #


def make_one_run_exp6(j, actions, mediators, rewards, w_threshold, B, singleW):
    n, T = actions.shape
    etaj_est_ours, _, _, pajts, pajt_1s = (
        estimate_mediation_j_effect_finite_multiple_stages_recursion(
            j=j,
            actions=actions,
            mediators=mediators,
            rewards=rewards,
            w_threshold=w_threshold,
            singleW=singleW,
            pajts=None,
            pajt_1s=None,
        )
    )
    q0_ours, q1_ours, bstd_ours = bootstrap_finite(
        estimator=estimate_mediation_j_effect_finite_multiple_stages_recursion_wrapper,
        j=j,
        B=B,
        actions=actions,
        mediators=mediators,
        rewards=rewards,
        w_threshold=w_threshold,
        singleW=singleW,
        pajts=pajts,
        pajt_1s=pajt_1s,
        cores=1,
    )
    etaj_est_conditional_independence = (
        estimate_mediation_j_effect_finite_multiple_stages_independence(
            j=j, actions=actions, mediators=mediators, rewards=rewards
        )
    )
    (
        q0_conditional_independence,
        q1_conditional_independence,
        bstd_conditional_independence,
    ) = bootstrap_finite(
        estimator=estimate_mediation_j_effect_finite_multiple_stages_independence,
        j=j,
        B=B,
        actions=actions,
        mediators=mediators,
        rewards=rewards,
        w_threshold=w_threshold,
        singleW=singleW,
        pajts=None,
        pajt_1s=None,
        cores=1,
    )
    etaj_est_time_independence, pajts = (
        estimate_mediation_j_effect_finite_multiple_stages_ignore(
            j=j,
            actions=actions,
            mediators=mediators,
            rewards=rewards,
            ancestor_t=True,
            pajts=None,
        )
    )
    (
        q0_time_independence,
        q1_time_independence,
        bstd_time_independence,
    ) = bootstrap_finite(
        estimator=estimate_mediation_j_effect_finite_multiple_stages_ignore_wrapper,
        j=j,
        B=B,
        actions=actions,
        mediators=mediators,
        rewards=rewards,
        w_threshold=w_threshold,
        singleW=singleW,
        pajts=pajts,
        pajt_1s=None,
        cores=1,
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
    # Js = [2]
    p_M = 0.9
    p_A = 0.5
    w_threshold = 0.1
    B = 100
    singleW = False

    suffix = datetime_suffix() + note

    res0 = expand_grid({"n": [100, 250, 500], "j": Js, "T": [10, 20, 30]})
    # res0 = expand_grid({"n": [100], "j": Js, "T": [10]})
    res = pd.DataFrame({})
    for idx, row in res0.iterrows():
        n, j, T = row["n"], row["j"], row["T"]
        t0 = time.time()
        mediators, actions, rewards, params_list = simulate_dataset_finite(
            n=row["n"], T=row["T"], d=d, p_M=p_M, p_A=p_A, seed=seed, nsim=nsim
        )
        print(params_list[0]["W0"])
        true_etaj = mc_true_eta_j(
            j=j, params_list=params_list, nsim_mc=100, n=100000, T=T
        )[-1]
        outs = p_map(
            make_one_run_exp6,
            [j] * nsim,
            [actions[i] for i in range(nsim)],
            [mediators[i] for i in range(nsim)],
            [rewards[i] for i in range(nsim)],
            [w_threshold] * nsim,
            [B] * nsim,
            [singleW] * nsim,
            num_cpus=CORES,
            desc="n:{},j:{},T:{}".format(n, j, T),
        )

        labels = ["ours", "conditional_independence", "time_independence"]
        t1 = time.time()
        for i in range(len(labels)):
            etajs_est = np.array([x[0][i][-1] for x in outs])
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
                    "est": etajs_est,
                    "true": [true_etaj] * nsim,
                    "bias": bias,
                    "q0": q0s,
                    "q1": q1s,
                    "bstd": bstds,
                    "covered": np.logical_and(
                        np.less_equal(q0s, true_etaj),
                        np.greater_equal(q1s, true_etaj),
                    ),
                    "singleW": [singleW] * nsim,
                    "time": [t1 - t0] * nsim,
                }
            )
            # df.to_csv(
            #     "./outs/tmp_finite_bs_exp6_d{}_seed{}_{}.csv".format(d, seed, suffix),
            #     index=False,
            #     header=False,
            #     mode="a",
            # )
            res = pd.concat([res, df])

    res.to_csv(
        "./outs/finite_bs_exp6_d{}_seed{}_{}.csv".format(d, seed, suffix), index=False
    )
    # print(res)


if __name__ == "__main__":
    run_exp6()
