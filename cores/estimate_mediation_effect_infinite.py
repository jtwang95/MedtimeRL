import numpy as np
import networkx as nx
from tqdm import tqdm
from .estimate_model_parameters import estimate_model_parameters
from p_tqdm import p_map


def estimate_mediation_j_effect_infinite(
    j, actions, mediators, rewards, params, **kwargs
):

    alpha1, beta1, Gamma1, zeta1, alpha2, beta2, zeta2, gamma2, kappa, W0 = (
        params["alpha1"],
        params["beta1"],
        params["Gamma1"],
        params["zeta1"],
        params["alpha2"],
        params["beta2"],
        params["zeta2"],
        params["gamma2"],
        params["kappa"],
        params["W0"],
    )
    n, T, d = mediators.shape
    G = nx.from_numpy_array(W0.T, create_using=nx.DiGraph)
    pajt = list(nx.ancestors(G, source=j))
    G = nx.from_numpy_array(
        np.block([[W0.T, Gamma1.T], [np.zeros([d, d]), W0.T]]), create_using=nx.DiGraph
    )
    pajt_1 = [x for x in list(nx.ancestors(G, source=j + d)) if x < d]

    # C1j M1j -> R1
    X = np.concatenate(
        [np.ones([n, T - 1, 1]), mediators[:, 1:T, [j]], actions[:, 1:T, np.newaxis]],
        axis=2,
    )
    X = np.concatenate([X, rewards[:, 0 : (T - 1), np.newaxis]], axis=2)  # Rt_1
    X = (
        np.concatenate([X, mediators[:, 0 : (T - 1), pajt_1]], axis=2)
        if len(pajt_1) > 0
        else X
    )  # Mt_1
    X = (
        np.concatenate([X, mediators[:, 1:T, pajt]], axis=2) if len(pajt) > 0 else X
    )  # Mt

    X = X.reshape([n * (T - 1), -1])
    Y = rewards[:, 1:T].reshape([-1, 1])
    C1j = np.linalg.solve(X.T @ X, X.T @ Y)[1, :]  # 1
    # print(C1j, ((np.identity(d) + W0.T) @ kappa)[j])
    # B1 M1j -> M1
    X = np.concatenate(
        [np.ones([n, T - 1, 1]), mediators[:, 1:T, [j]], actions[:, 1:T, np.newaxis]],
        axis=2,
    )
    X = np.concatenate([X, rewards[:, 0 : (T - 1), np.newaxis]], axis=2)  # Rt_1
    X = (
        np.concatenate([X, mediators[:, 0 : (T - 1), pajt_1]], axis=2)
        if len(pajt_1) > 0
        else X
    )  # Mt_1
    X = (
        np.concatenate([X, mediators[:, 1:T, pajt]], axis=2) if len(pajt) > 0 else X
    )  # Mt

    X = X.reshape([n * (T - 1), -1])
    Y = mediators[:, 1:T, :].reshape([-1, d])
    B1 = np.linalg.solve(X.T @ X, X.T @ Y)[1, :]  # 1
    # print(B1, (np.identity(d) + W0.T)[j, :])

    # A_tilde
    # print((zeta1.reshape(-1, 1) @ (kappa + gamma2).reshape(1, -1)).shape)
    X = (1 - zeta2) * (np.identity(d) - Gamma1) - zeta1.reshape(-1, 1) @ (
        kappa + gamma2
    ).reshape(1, -1)
    Y = (1 - zeta2) * beta1 + beta2 * zeta1
    A_tilde = np.linalg.solve(X, Y).reshape(-1)
    # print("calculated A_tilde_j:{}".format(A_tilde[j]))

    # C_tilde
    X = (
        1
        - zeta2
        - (kappa + gamma2).reshape(1, -1)
        @ np.linalg.solve(np.identity(d) - Gamma1, zeta1)
    )
    Y = (zeta2 * C1j + gamma2.reshape(1, -1) @ B1) * A_tilde[j]
    Y += (kappa + gamma2).reshape(1, -1) @ np.linalg.solve(
        np.identity(d) - Gamma1, (Gamma1 @ B1 + zeta1 * C1j) * A_tilde[j]
    )
    C_tilde = (Y / X).flatten()

    # B_tilde
    X = np.identity(d) - Gamma1
    # print((Gamma1 @ B1).shape, (C_tilde).shape)
    Y = (Gamma1 @ B1 + zeta1 * C1j) * A_tilde[j] + zeta1 * C_tilde
    B_tilde = np.linalg.solve(X, Y).reshape(-1)

    # B_tilde, C_tilde matrix version
    U = np.block(
        [[np.zeros([d, d]), np.zeros([d, 1])], [kappa.reshape(1, -1), np.zeros([1, 1])]]
    )
    V = np.block(
        [[Gamma1, zeta1.reshape(d, 1)], [gamma2.reshape(1, -1), zeta2.reshape(1, 1)]]
    )
    X = np.identity(d + 1) - U - V
    Y = V @ np.concatenate([B1, C1j.reshape(1)], axis=0) * A_tilde[j]
    S = np.linalg.solve(X, Y)
    B_tilde_matrix = S.squeeze()[:d]
    C_tilde_matrix = S.squeeze()[d]
    # B_tilde_diff = np.linalg.norm(B_tilde - B_tilde_matrix)
    # C_tilde_diff = np.linalg.norm(C_tilde - C_tilde_matrix)

    # B_bar new version
    U = np.block(
        [[np.zeros([d, d]), np.zeros([d, 1])], [kappa.reshape(1, -1), np.zeros([1, 1])]]
    )
    V = np.block(
        [[Gamma1, zeta1.reshape(d, 1)], [gamma2.reshape(1, -1), zeta2.reshape(1, 1)]]
    )
    G1 = np.concatenate([B1, C1j.reshape(1)], axis=0)
    S = np.linalg.solve(np.identity(d + 1) - U - V, V @ G1)
    B_bar = S.squeeze()[:d]

    # I1 and I2
    I1 = A_tilde[j]
    I2 = (C_tilde - C1j * B_tilde[j]) / (1 + B_bar[j])
    # print("calculated C1j:{}".format(C1j))
    debug = False
    if debug:
        print("calculated I1:{}".format(I1))
        print("calculated I2:{}".format(I2))
    # print("calculated undivided I2:{}".format(I2 * (1 + B_bar[j])))
    # print(B1)

    # etaj
    etaj = (C1j * I1 + I2)[0]
    return etaj


def bootstrap_infinite(
    estimator,
    j,
    B,
    actions,
    mediators,
    rewards,
    cores,
    w_threshold=0.3,
    W0=None,
    paj=None,
    flag_estimate_W0=True,
):
    n, T, d = mediators.shape
    if cores == 1:
        etajsb = np.zeros([B])
        for b in tqdm(range(B), desc="bootstrap"):
            idx = np.random.choice(n, n, replace=True)
            actions_b = actions[idx]
            mediators_b = mediators[idx]
            rewards_b = rewards[idx]
            params_est = estimate_model_parameters(
                actions=actions_b,
                mediators=mediators_b,
                rewards=rewards_b,
                estimate_W0=flag_estimate_W0,
                estimate_W0_lambda1=0.00,
                estimate_W0_threshold=w_threshold,
            )
            if not flag_estimate_W0:
                params_est["W0"] = W0
            tmp = estimator(
                j=j,
                actions=actions_b,
                mediators=mediators_b,
                rewards=rewards_b,
                params=params_est,
                paj=paj,
            )
            etajsb[b] = tmp
    else:

        def run_one(j, actions, mediators, rewards, w_threshold):
            params_est = estimate_model_parameters(
                actions=actions,
                mediators=mediators,
                rewards=rewards,
                estimate_W0=flag_estimate_W0,
                estimate_W0_lambda1=0.00,
                estimate_W0_threshold=w_threshold,
            )
            if not flag_estimate_W0:
                params_est["W0"] = W0
            return estimator(
                j=j,
                actions=actions,
                mediators=mediators,
                rewards=rewards,
                params=params_est,
                paj=paj,
            )

        idxs = np.random.choice(n, n * B, replace=True).reshape([B, n])
        tmp = p_map(
            run_one,
            [j] * B,
            [actions[idxs[b]] for b in range(B)],
            [mediators[idxs[b]] for b in range(B)],
            [rewards[idxs[b]] for b in range(B)],
            [w_threshold] * B,
            num_cpus=cores,
            desc="bootstrap",
        )
        etajsb = np.array([x for x in tmp])

    q0 = np.percentile(etajsb, 2.5)
    q1 = np.percentile(etajsb, 97.5)
    return q0, q1, np.std(etajsb)


if __name__ == "__main__":
    pass
