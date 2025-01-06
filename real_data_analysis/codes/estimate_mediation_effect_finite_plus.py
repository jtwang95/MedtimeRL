import sys

sys.path.append("../../cores")

import numpy as np
from p_tqdm import p_map
from tqdm import tqdm

from estimate_mediation_effect_finite import (
    estimate_mediation_j_effect_finite_multiple_stages_recursion,
)


def estimate_mediation_j_effect_finite_multiple_stages_recursion_wrapper(
    j, actions, mediators, rewards, w_threshold, singleW
):
    """
    for bootstrap evaluation
    """
    out = estimate_mediation_j_effect_finite_multiple_stages_recursion(
        j=j,
        actions=actions,
        mediators=mediators,
        rewards=rewards,
        w_threshold=w_threshold,
        singleW=singleW,
    )
    if singleW:
        etaj, etaj_immediate, etaj_lagged, _ = out
    else:
        etaj, etaj_immediate, etaj_lagged = out
    return etaj, etaj_immediate, etaj_lagged


def bootstrap_finite(
    estimator, j, B, actions, mediators, rewards, w_threshold, singleW, cores
):
    n, T, d = mediators.shape
    if cores == 1:
        etajs_b = np.zeros([B, T])
        etajs_immediate_b = np.zeros([B, T])
        etajs_lagged_b = np.zeros([B, T])
        for b in tqdm(range(B), desc="bootstrap"):
            idx = np.random.choice(n, n, replace=True)
            actions_b = actions[idx]
            mediators_b = mediators[idx]
            rewards_b = rewards[idx]
            etaj_b, etaj_immediate_b, etaj_lagged_b = estimator(
                j=j,
                actions=actions_b,
                mediators=mediators_b,
                rewards=rewards_b,
                w_threshold=w_threshold,
                singleW=singleW,
            )
            etajs_b[b] = etaj_b
            etajs_immediate_b[b] = etaj_immediate_b
            etajs_lagged_b[b] = etaj_lagged_b
    else:
        idxs = np.random.choice(n, n * B, replace=True).reshape([B, n])
        tmp = p_map(
            estimator,
            [j] * B,
            [actions[idxs[b]] for b in range(B)],
            [mediators[idxs[b]] for b in range(B)],
            [rewards[idxs[b]] for b in range(B)],
            [w_threshold] * B,
            [singleW] * B,
            num_cpus=cores,
            desc="bootstrap",
        )
        etajs_b = np.vstack([x[0] for x in tmp])
        etajs_immediate_b = np.vstack([x[1] for x in tmp])
        etajs_lagged_b = np.vstack([x[2] for x in tmp])

    etajs_q0 = np.percentile(etajs_b, 2.5, axis=0)
    etajs_q1 = np.percentile(etajs_b, 97.5, axis=0)
    etajs_std = np.std(etajs_b, axis=0)

    etajs_immediate_q0 = np.percentile(etajs_immediate_b, 2.5, axis=0)
    etajs_immediate_q1 = np.percentile(etajs_immediate_b, 97.5, axis=0)
    etajs_immediate_std = np.std(etajs_immediate_b, axis=0)

    etajs_lagged_q0 = np.percentile(etajs_lagged_b, 2.5, axis=0)
    etajs_lagged_q1 = np.percentile(etajs_lagged_b, 97.5, axis=0)
    etajs_lagged_std = np.std(etajs_lagged_b, axis=0)
    return (
        etajs_q0,
        etajs_q1,
        etajs_std,
        etajs_immediate_q0,
        etajs_immediate_q1,
        etajs_immediate_std,
        etajs_lagged_q0,
        etajs_lagged_q1,
        etajs_lagged_std,
    )


if __name__ == "__main__":
    pass
