import numpy as np
import networkx as nx
from dagma.linear import DagmaLinear
from p_tqdm import p_map
from tqdm import tqdm


def estimate_parent(
    j,
    at,
    mt,
    ancestor_t,
    mt_1=None,
    rt_1=None,
    ancestor_t1=None,
    w_lambda1=0.00,
    w_threshold=0.3,
):
    n, d = mt.shape
    intercept = np.ones([n, 1])
    Mt_flat = mt.reshape([n, -1])
    if mt_1 is None:
        Xt = np.concatenate([intercept, at], axis=1).reshape([n, -1])
        Theta1_hat = np.linalg.solve(Xt.T @ Xt, Xt.T @ Mt_flat)
    else:
        Xt = np.concatenate([intercept, at, mt_1, rt_1], axis=1).reshape([n, -1])
        Theta1_hat = np.linalg.solve(Xt.T @ Xt, Xt.T @ Mt_flat)
        Gamma1 = Theta1_hat[2 : (d + 2), :].T

    model = DagmaLinear(loss_type="l2")
    # eq_model = DagmaMLP(dims=[d, 10, 1], bias=True, dtype=torch.double)
    # model = DagmaNonlinear(eq_model, dtype=torch.double)
    W_est = model.fit(
        Mt_flat - Xt @ Theta1_hat,
        lambda1=w_lambda1,
        w_threshold=w_threshold,
        warm_iter=400,
        max_iter=1000,
        T=6,
    )
    # print(W_est.T)
    G = nx.from_numpy_array(W_est, create_using=nx.DiGraph)
    if ancestor_t:
        pajt = list(nx.ancestors(G, source=j))
    else:
        pajt = list(G.predecessors(j))

    if ancestor_t1 is not None:
        G = nx.from_numpy_array(
            np.block([[W_est, Gamma1.T], [np.zeros([d, d]), W_est]]),
            create_using=nx.DiGraph,
        )
        if ancestor_t1:
            pajt_1 = [x for x in list(nx.ancestors(G, source=j + d)) if x < d]
        else:
            pajt_1 = [x for x in list(G.predecessors(j + d)) if x < d]
        return pajt, pajt_1, W_est
    else:
        return pajt, W_est


def estimate_stagewise_model_parameters(
    action_t,
    mediator_t,
    reward_t,
    mediator_t1,
    reward_t1,
    estimate_W0=False,
    estimate_W0_lambda1=0.02,
):
    n, d = mediator_t.shape

    intercept = np.ones([n, 1])
    At = action_t[:, np.newaxis]
    Mt = mediator_t
    Rt = reward_t[:, np.newaxis]

    if mediator_t1 is not None:
        Mt_1 = mediator_t1
        Rt_1 = reward_t1[:, np.newaxis]
        Xt = np.concatenate([intercept, At, Mt_1, Rt_1], axis=1).reshape([-1, d + 3])
        Zt = np.concatenate([intercept, At, Rt_1, Mt_1, Mt], axis=1).reshape(
            [-1, 2 * d + 3]
        )
        Mt_flat = Mt.reshape([-1, d])
        Rt_flat = Rt.reshape([-1, 1])
        Theta1_hat = np.linalg.solve(Xt.T @ Xt, Xt.T @ Mt_flat)
        Theta2_hat = np.linalg.solve(Zt.T @ Zt, Zt.T @ Rt_flat)
        alpha1_hat, beta1_hat, Gamma1_hat, zeta1_hat = (
            Theta1_hat[0, :],
            Theta1_hat[1, :],
            Theta1_hat[2 : (d + 2), :].T,
            Theta1_hat[d + 2, :],
        )
        alpha2_hat, beta2_hat, zeta2_hat, gamma2_hat, kappa_hat = (
            Theta2_hat[0, :].flatten(),
            Theta2_hat[1, :].flatten(),
            Theta2_hat[2, :].flatten(),
            Theta2_hat[3 : (d + 3), :].flatten(),
            Theta2_hat[(d + 3) : (2 * d + 3), :].flatten(),
        )
    else:
        Xt = np.concatenate([intercept, At], axis=2).reshape([-1, 2])
        Zt = np.concatenate([intercept, At, Mt], axis=2).reshape([-1, d + 2])
        Mt_flat = Mt.reshape([-1, d])
        Rt_flat = Rt.reshape([-1, 1])
        Theta1_hat = np.linalg.solve(Xt.T @ Xt, Xt.T @ Mt_flat)
        Theta2_hat = np.linalg.solve(Zt.T @ Zt, Zt.T @ Rt_flat)
        alpha1_hat, beta1_hat = Theta1_hat[0, :], Theta1_hat[1, :]
        alpha2_hat, beta2_hat, kappa_hat = (
            Theta2_hat[0, :].flatten(),
            Theta2_hat[1, :].flatten(),
            Theta2_hat[2:, :].flatten(),
        )
        Gamma1_hat = np.nan
        zeta1_hat = np.nan
        zeta2_hat = np.nan
        gamma2_hat = np.nan

    params_est = {
        "alpha1": alpha1_hat,
        "beta1": beta1_hat,
        "Gamma1": Gamma1_hat,
        "zeta1": zeta1_hat,
        "alpha2": alpha2_hat,
        "beta2": beta2_hat,
        "zeta2": zeta2_hat,
        "gamma2": gamma2_hat,
        "kappa": kappa_hat,
    }

    if estimate_W0:
        model = DagmaLinear(loss_type="l2")
        # eq_model = DagmaMLP(dims=[d, 32, 1], bias=True, dtype=torch.double)
        # model = DagmaNonlinear(eq_model, dtype=torch.double)
        W_est = model.fit(
            Mt_flat - Xt @ Theta1_hat, lambda1=estimate_W0_lambda1, max_iter=10000
        )
        params_est["W0"] = W_est.T

    return params_est


def estimate_mediation_j_effect_finite_multiple_stages_recursion(
    j, actions, mediators, rewards, w_threshold, singleW=False, pajts=None, pajt_1s=None
):
    if pajts is None and pajt_1s is None:
        flag_need_pajt_pajt_1 = True
    else:
        flag_need_pajt_pajt_1 = False
    is_ancestor_t = True
    is_ancestor_t1 = True
    n, T, d = mediators.shape
    etaj = np.zeros([T])
    etaj_immediate = np.zeros([T])
    etaj_lagged = np.zeros([T])
    sumAj = []
    Bj = []
    ancestors_t = {}
    ancestors_t1 = {}
    # A1
    X = np.concatenate([np.ones([n, 1]), actions[:, [0]]], axis=1)
    Y = mediators[:, 0, :]
    A1 = np.linalg.solve(X.T @ X, X.T @ Y)[1, :]
    # Cj1
    X = np.concatenate([np.ones([n, 1]), mediators[:, 0, [j]], actions[:, [0]]], axis=1)
    if flag_need_pajt_pajt_1:
        pajt, _ = estimate_parent(
            j=j,
            at=actions[:, [0]],
            mt=mediators[:, 0, :],
            ancestor_t=is_ancestor_t,
            w_threshold=w_threshold,
        )
        pajts = [pajt]
        pajt_1s = []
    else:
        pajt = pajts[0]
    ancestors_t[0] = pajt
    X = np.concatenate([X, mediators[:, 0, pajt]], axis=1) if len(pajt) > 0 else X
    X = X.reshape([n, -1])
    Y = rewards[:, 0].reshape([-1, 1])
    Cj1 = np.linalg.solve(X.T @ X, X.T @ Y)[1, :]
    # B1
    X = np.concatenate([np.ones([n, 1]), mediators[:, 0, [j]], actions[:, [0]]], axis=1)
    pajt = ancestors_t[0]
    X = X.reshape([n, -1])
    Y = mediators[:, 0, :].reshape([-1, d])
    B1 = np.linalg.solve(X.T @ X, X.T @ Y)[1, :]
    # D1
    X = np.concatenate([np.ones([n, 1]), actions[:, [0]]], axis=1)
    Y = rewards[:, 0].reshape([-1, 1])
    D1 = np.linalg.solve(X.T @ X, X.T @ Y)[1, :].squeeze()
    # etaj1
    etaj[0] = A1[j] * Cj1[0]
    etaj_immediate[0] = etaj[0]
    etaj_lagged[0] = 0.0
    sumAj.append(np.sum(A1[j]))

    At_1 = A1.reshape(1, d)
    Bt_1 = B1.reshape(1, d)
    Cjt_1 = Cj1.reshape(1)
    Dt_1 = D1.reshape(1)

    if singleW:
        pajt, pajt_1, W_est = estimate_parent(
            j=j,
            at=actions[:, 1:T].reshape(n * (T - 1), 1),
            mt=mediators[:, 1:T, :].reshape(n * (T - 1), d),
            mt_1=mediators[:, 0 : (T - 1), :].reshape(n * (T - 1), d),
            rt_1=rewards[:, 1:T].reshape(n * (T - 1), 1),
            ancestor_t=is_ancestor_t,
            ancestor_t1=is_ancestor_t1,
            w_threshold=w_threshold,
        )

    for t in range(1, T):
        At = np.zeros([t + 1, d])
        Bt = np.zeros([t + 1, d])
        Cjt = np.zeros([t + 1])
        Dt = np.zeros([t + 1])

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
        if not singleW and flag_need_pajt_pajt_1:
            pajt, pajt_1, _ = estimate_parent(
                j=j,
                at=actions[:, [t]],
                mt=mediators[:, t, :],
                mt_1=mediators[:, t - 1, :],
                rt_1=rewards[:, [t - 1]],
                ancestor_t=is_ancestor_t,
                ancestor_t1=is_ancestor_t1,
                w_threshold=w_threshold,
            )
            pajts.append(pajt)
            pajt_1s.append(pajt_1)
        elif not singleW and not flag_need_pajt_pajt_1:
            pajt = pajts[t]
            pajt_1 = pajt_1s[t - 1]
        ancestors_t[t] = pajt
        ancestors_t1[t] = pajt_1
        # estimate A_{1}^{(t)}, D_{1}^{(t)},B_{1}^{(t)},C_{1j}^{(t)} using regression
        ## A_{1}^{(t)}
        X = np.concatenate([np.ones([n, 1]), actions[:, [t]]], axis=1)
        Y = mediators[:, t, :]
        At[0] = np.linalg.solve(X.T @ X, X.T @ Y)[1, :]
        ## D_{1}^{(t)}
        X = np.concatenate([np.ones([n, 1]), actions[:, [t]]], axis=1)
        Y = rewards[:, t].reshape([-1, 1])
        Dt[0] = np.linalg.solve(X.T @ X, X.T @ Y)[1, :].squeeze()
        ## B_{1}^{(t)}
        X = np.concatenate(
            [
                np.ones([n, 1]),
                mediators[:, t, [j]],
                actions[:, [t]],
                rewards[:, [t - 1]],
            ],
            axis=1,
        )
        X = (
            np.concatenate([X, mediators[:, t - 1, pajt_1]], axis=1)
            if len(pajt_1) > 0
            else X
        )
        X = np.concatenate([X, mediators[:, t, pajt]], axis=1) if len(pajt) > 0 else X
        X = X.reshape([n, -1])
        Y = mediators[:, t, :].reshape([-1, d])
        Bt[0] = np.linalg.solve(X.T @ X, X.T @ Y)[1, :]
        ## C_{1j}^{(t)}
        X = np.concatenate(
            [
                np.ones([n, 1]),
                mediators[:, t, [j]],
                actions[:, [t]],
                rewards[:, [t - 1]],
            ],
            axis=1,
        )
        X = (
            np.concatenate([X, mediators[:, t - 1, pajt_1]], axis=1)
            if len(pajt_1) > 0
            else X
        )
        X = np.concatenate([X, mediators[:, t, pajt]], axis=1) if len(pajt) > 0 else X
        X = X.reshape([n, -1])
        Y = rewards[:, t].reshape([-1, 1])
        Cjt[0] = np.linalg.solve(X.T @ X, X.T @ Y)[1, :].squeeze()

        # estimate the rest of A_{i}^{(t)}, D_{i}^{(t)},B_{i}^{(t)},C_{ij}^{(t)} for i=2,...,t using recursion
        for tt in range(1, t + 1):
            At[tt] = (
                Gamma1 @ At_1[tt - 1].reshape(-1, 1)
                + zeta1.reshape(-1, 1) * Dt_1[tt - 1]
            ).squeeze()
            Dt[tt] = (
                zeta2 * Dt_1[tt - 1]
                + At[tt].reshape(1, -1) @ kappa.reshape(-1, 1)
                + At_1[tt - 1].reshape(1, -1) @ gamma2
            )
            Bt[tt] = (
                Gamma1 @ Bt_1[tt - 1].reshape(-1, 1)
                + zeta1.reshape(-1, 1) * Cjt_1[tt - 1]
            ).squeeze()
            Cjt[tt] = (
                zeta2 * Cjt_1[tt - 1]
                + kappa.reshape(1, -1) @ Bt[tt].reshape(-1, 1)
                + gamma2.reshape(1, -1) @ Bt_1[tt - 1].reshape(-1, 1)
            )

        # update sumA and Bj
        sumAj.append(np.sum(At[:, j]))
        for tt in range(t - 1):
            Bj[tt].append(Bt[tt + 1, j])
        Bj.append([Bt[t, j]])

        ## S_{1j}^{(t)} - S_{(t-1)j}^{(t)}
        Sjt = np.zeros([t])
        svec = [Cjt[0]]
        for tt in range(t):
            bvec = np.array([Bj[ttt][t - tt - 1] for ttt in range(tt + 1)])[::-1]
            # print(bvec.shape, np.array(svec).shape)
            Sjt[tt] = Cjt[tt + 1] - np.sum(bvec * np.array(svec))
            svec.append(Sjt[tt])

        # calculate etajt
        svec = np.concatenate([np.array([Cjt[0]]), Sjt], axis=0)
        etaj[t] = t * etaj[t - 1] + np.sum(np.array(sumAj)[::-1] * svec)
        etaj[t] /= t + 1
        etaj_immediate[t] = Cjt[0] * At[0, j]
        etaj_lagged[t] = np.sum(np.array(sumAj)[::-1] * svec) - etaj_immediate[t]

        # update At_1,Bt_1,Cjt_1,Dt_1
        At_1 = At
        Bt_1 = Bt
        Cjt_1 = Cjt
        Dt_1 = Dt
        # print("recursion:{}".format(At[:, j]))

    # times t
    etaj = etaj * np.arange(1, T + 1)
    etaj_immediate = etaj_immediate * np.arange(1, T + 1)
    etaj_lagged = etaj_lagged * np.arange(1, T + 1)

    if singleW:
        return etaj, etaj_immediate, etaj_lagged, W_est
    else:
        return etaj, etaj_immediate, etaj_lagged, pajts, pajt_1s


def estimate_mediation_j_effect_finite_multiple_stages_recursion_wrapper(
    j, actions, mediators, rewards, w_threshold, singleW, pajts, pajt_1s
):
    """
    for bootstrap evaluation
    """
    return estimate_mediation_j_effect_finite_multiple_stages_recursion(
        j=j,
        actions=actions,
        mediators=mediators,
        rewards=rewards,
        w_threshold=w_threshold,
        singleW=singleW,
        pajts=pajts,
        pajt_1s=pajt_1s,
    )[0]


def bootstrap_finite(
    estimator,
    j,
    B,
    actions,
    mediators,
    rewards,
    w_threshold,
    singleW,
    pajts,
    pajt_1s,
    cores,
):
    n, T, d = mediators.shape
    if cores == 1:
        etajsb = np.zeros([B])
        for b in tqdm(range(B), desc="bootstrap"):
            idx = np.random.choice(n, n, replace=True)
            actions_b = actions[idx]
            mediators_b = mediators[idx]
            rewards_b = rewards[idx]
            tmp = estimator(
                j=j,
                actions=actions_b,
                mediators=mediators_b,
                rewards=rewards_b,
                w_threshold=w_threshold,
                singleW=singleW,
                pajts=pajts,
                pajt_1s=pajt_1s,
            )
            etajsb[b] = tmp[-1]
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
            [pajts] * B,
            [pajt_1s] * B,
            num_cpus=cores,
            desc="bootstrap",
        )
        etajsb = np.array([x[-1] for x in tmp])

    q0 = np.percentile(etajsb, 2.5)
    q1 = np.percentile(etajsb, 97.5)
    return q0, q1, np.std(etajsb)


if __name__ == "__main__":
    pass
