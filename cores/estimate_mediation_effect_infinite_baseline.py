import numpy as np
import networkx as nx
from dagma.linear import DagmaLinear


def estimate_mediation_j_effect_infinite_conditional_independence(
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
    W0 = np.zeros([d, d])
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

    # A_tilde
    X = (1 - zeta2) * (np.identity(d) - Gamma1) - zeta1.reshape(-1, 1) @ (
        kappa + gamma2
    ).reshape(1, -1)
    Y = (1 - zeta2) * beta1 + beta2 * zeta1
    A_tilde = np.linalg.solve(X, Y).reshape(-1)

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
    debug = False
    if debug:
        print("calculated I1:{}".format(I1))
        print("calculated I2:{}".format(I2))

    # etaj
    etaj = (C1j * I1 + I2)[0]
    return etaj


def estimate_mediation_j_effect_infinite_ignore_time_dependence(
    j, actions, mediators, rewards, paj=None, **kwargs
):
    if paj is None:
        flag_need_paj = True
    else:
        flag_need_paj = False

    def estimate_parent(j, at, mt, ancestor_t, estimate_W0_lambda1=0.00):
        n, d = mt.shape
        intercept = np.ones([n, 1])
        Mt_flat = mt.reshape([n, -1])
        Xt = np.concatenate([intercept, at], axis=1).reshape([n, -1])
        Theta1_hat = np.linalg.solve(Xt.T @ Xt, Xt.T @ Mt_flat)

        model = DagmaLinear(loss_type="l2")
        W_est = model.fit(Mt_flat - Xt @ Theta1_hat, lambda1=estimate_W0_lambda1)
        G = nx.from_numpy_array(W_est, create_using=nx.DiGraph)
        if ancestor_t:
            pajt = list(nx.ancestors(G, source=j))
        else:
            pajt = list(G.predecessors(j))
        return pajt

    n, T, d = mediators.shape
    is_ancestor = True
    if flag_need_paj:
        paj = estimate_parent(
            j=j,
            at=actions[:, :].reshape(-1, 1),
            mt=mediators[:, :, :].reshape(-1, d),
            ancestor_t=is_ancestor,
        )
    else:
        paj = paj
    # A_t A_t -> M_t
    X = np.concatenate([np.ones([n, T, 1]), actions[:, :, np.newaxis]], axis=2)
    X = X.reshape([n * T, -1])
    Y = mediators[:, :, [j]].reshape([n * T, -1])
    Aj = np.linalg.solve(X.T @ X, X.T @ Y)[1, :]

    # C_t M_t -> R_t
    X = np.concatenate(
        [np.ones([n, T, 1]), mediators[:, :, [j]], actions[:, :, np.newaxis]], axis=2
    )
    X = np.concatenate([X, mediators[:, :, paj]], axis=2) if len(paj) > 0 else X
    X = X.reshape([n * T, -1])
    Y = rewards[:, :].reshape([-1, 1])
    Cj = np.linalg.solve(X.T @ X, X.T @ Y)[1, :]

    # calculate etaj
    etaj = (Aj * Cj)[0]

    return etaj, paj


def estimate_mediation_j_effect_infinite_ignore_time_dependence_wrapper(
    j, actions, mediators, rewards, paj=None, **kwargs
):
    return estimate_mediation_j_effect_infinite_ignore_time_dependence(
        j=j, actions=actions, mediators=mediators, rewards=rewards, paj=paj, **kwargs
    )[0]


if __name__ == "__main__":
    pass
