import numpy as np
import copy


def sample_from_conditional_mvn(cond_idx, cond_val, mus, Cov):
    """
    mus: n*d
    Sigma: d*d
    cond_idx: 0<= . <= d-1
    cond_val: double
    """
    n, d = mus.shape
    j = cond_idx
    M = np.zeros([n, d])
    ## idx < j
    if j >= 1:
        mu_j = mus[:, :j]
        Cov_j = Cov[:j, :j]
        # https://stackoverflow.com/questions/69399035/is-there-a-way-of-batch-sampling-from-numpys-multivariate-normal-distribution-i
        L = np.linalg.cholesky(Cov_j)
        M_j = (L @ np.random.normal(0, 1, size=[j, n])).T + mu_j
        M[:, :j] = M_j
    ## idx == j
    Mj = np.ones([n, 1]) * cond_val
    M[:, j] = Mj.flatten()

    ## idx > j
    if j <= d - 2:
        C11 = Cov[: (j + 1), : (j + 1)].reshape([j + 1, j + 1])
        C21 = Cov[(j + 1) :, : (j + 1)].reshape([d - j - 1, j + 1])
        C12 = Cov[: (j + 1), (j + 1) :].reshape([j + 1, d - j - 1])
        C22 = Cov[(j + 1) :, (j + 1) :].reshape([d - j - 1, d - j - 1])

        muj_marginal = mus[:, (j + 1) :]
        if j >= 1:
            m = np.concatenate([M_j, Mj], axis=1)
        else:
            m = Mj
        muj_ = muj_marginal + (C21 @ np.linalg.solve(C11, m.T - mus[:, : (j + 1)].T)).T

        Covj_ = C22 - C21 @ np.linalg.solve(C11, C12)
        L = np.linalg.cholesky(Covj_)

        Mj_ = (L @ np.random.normal(0, 1, size=[d - j - 1, n])).T + muj_

        M[:, (j + 1) :] = Mj_
    return M


def calculate_true_etaj_infinite(j, params, warmup=5):
    (
        n,
        T,
        sigma_m,
        sigma_r,
        alpha1,
        beta1,
        Gamma1,
        zeta1,
        alpha2,
        beta2,
        zeta2,
        gamma2,
        kappa,
        W0,
    ) = (
        params["n"],
        params["T"],
        params["sigma_m"],
        params["sigma_r"],
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
    d = beta1.shape[0]
    # init
    etajs = np.zeros([T])
    # _ = np.repeat(0, n)
    Rt1_1, Rt0_1, Rt1_1_Mj, Rt0_1_Mj = (
        np.repeat(0, n),
        np.repeat(0, n),
        np.repeat(0, n),
        np.repeat(0, n),
    )  # Rt1_1 = R_{t-1} | A_i=1 for i <= (t-1)
    Mt1_1 = np.zeros([n, d])
    Mt0_1 = np.zeros([n, d])
    Mt1_1_Mj = np.zeros([n, d])
    Mt0_1_Mj = np.zeros([n, d])
    c = 5
    # print(kappa)
    Mt1_1_Mj[:, j], Mt0_1_Mj[:, j] = c, c
    Rt_delta, Rt_Mj_delta = np.zeros([T]), np.zeros([T])
    I_W0_inv = np.linalg.inv(np.identity(d) - W0)
    Cov_Mt = I_W0_inv @ I_W0_inv.T
    # print(Cov_Mt)

    for t in range(T + warmup):
        At0 = np.repeat(0, n)
        At1 = np.repeat(1, n)
        epsilon_t = np.random.normal(0, sigma_m, size=[n, d])

        Mt1 = (
            np.linalg.solve(np.identity(d) - W0, epsilon_t.T).T
            + alpha1
            + (Gamma1 @ Mt1_1.T).T
            + At1[:, np.newaxis] @ beta1[np.newaxis, :]
            + Rt1_1[:, np.newaxis] @ zeta1[np.newaxis, :]
        )
        Mt0 = (
            np.linalg.solve(np.identity(d) - W0, epsilon_t.T).T
            + alpha1
            + (Gamma1 @ Mt0_1.T).T
            + At0[:, np.newaxis] @ beta1[np.newaxis, :]
            + Rt0_1[:, np.newaxis] @ zeta1[np.newaxis, :]
        )

        mu1 = (
            alpha1
            + (Gamma1 @ Mt1_1_Mj.T).T
            + At1[:, np.newaxis] @ beta1[np.newaxis, :]
            + Rt1_1_Mj[:, np.newaxis] @ zeta1[np.newaxis, :]
        )
        Cov1 = Cov_Mt
        Mt1_Mj = sample_from_conditional_mvn(cond_idx=j, cond_val=c, mus=mu1, Cov=Cov1)

        mu0 = (
            alpha1
            + (Gamma1 @ Mt0_1_Mj.T).T
            + At0[:, np.newaxis] @ beta1[np.newaxis, :]
            + Rt0_1_Mj[:, np.newaxis] @ zeta1[np.newaxis, :]
        )
        Cov0 = Cov_Mt
        Mt0_Mj = sample_from_conditional_mvn(cond_idx=j, cond_val=c, mus=mu0, Cov=Cov0)

        mu_R1 = (
            alpha2
            + beta2 * At1
            + zeta2 * Rt1_1
            + (Mt1_1 @ gamma2).squeeze()
            + (Mt1 @ kappa).squeeze()
        )
        mu_R0 = (
            alpha2
            + beta2 * At0
            + zeta2 * Rt0_1
            + (Mt0_1 @ gamma2).squeeze()
            + (Mt0 @ kappa).squeeze()
        )

        mu_R1_Mj = (
            alpha2
            + beta2 * At1
            + zeta2 * Rt1_1_Mj
            + (Mt1_1_Mj @ gamma2).squeeze()
            + (Mt1_Mj @ kappa).squeeze()
        )
        mu_R0_Mj = (
            alpha2
            + beta2 * At0
            + zeta2 * Rt0_1_Mj
            + (Mt0_1_Mj @ gamma2).squeeze()
            + (Mt0_Mj @ kappa).squeeze()
        )
        # print(np.mean(Mt1_Mj - Mt0_Mj, axis=0))

        Rt1 = np.random.normal(loc=mu_R1, scale=sigma_r)
        Rt0 = np.random.normal(loc=mu_R0, scale=sigma_r)

        Rt1_Mj = np.random.normal(loc=mu_R1_Mj, scale=sigma_r)
        Rt0_Mj = np.random.normal(loc=mu_R0_Mj, scale=sigma_r)

        Mt1_1 = copy.deepcopy(Mt1)
        Mt0_1 = copy.deepcopy(Mt0)
        Rt1_1 = copy.deepcopy(Rt1)
        Rt0_1 = copy.deepcopy(Rt0)
        Mt1_1_Mj = copy.deepcopy(Mt1_Mj)
        Mt0_1_Mj = copy.deepcopy(Mt0_Mj)
        Rt1_1_Mj = copy.deepcopy(Rt1_Mj)
        Rt0_1_Mj = copy.deepcopy(Rt0_Mj)
        if t < warmup:
            Mt1_1 = copy.deepcopy((Mt1 + Mt0) / 2)
            Mt0_1 = copy.deepcopy((Mt1 + Mt0) / 2)
            Rt1_1 = copy.deepcopy((Rt1 + Rt0) / 2)
            Rt0_1 = copy.deepcopy((Rt1 + Rt0) / 2)
            Mt1_1_Mj = copy.deepcopy((Mt1 + Mt0) / 2)
            Mt0_1_Mj = copy.deepcopy((Mt1 + Mt0) / 2)
            Rt1_1_Mj = copy.deepcopy((Rt1 + Rt0) / 2)
            Rt0_1_Mj = copy.deepcopy((Rt1 + Rt0) / 2)
            # pass
        if t >= warmup:
            tt = t - warmup
            Rt_delta[tt] = np.mean(Rt1 - Rt0)
            Rt_Mj_delta[tt] = np.mean(Rt1_Mj - Rt0_Mj)
            etajs[tt] = np.mean(Rt_delta[: (tt + 1)]) - np.mean(Rt_Mj_delta[: (tt + 1)])
    return etajs[-1]


def calculate_true_etaj_finite(j, n, T, params_list):
    d, W0 = params_list[0]["d"], params_list[0]["W0"]
    # init
    etajs = np.zeros([T])
    c = 5
    Rt_delta, Rt_Mj_delta = np.zeros([T]), np.zeros([T])
    I_W0_inv = np.linalg.inv(np.identity(d) - W0)
    Cov_Mt = I_W0_inv @ I_W0_inv.T

    for t in range(T):
        params = params_list[t]
        (
            p_M,
            p_A,
            sigma_m,
            sigma_r,
            alpha1_t,
            beta1_t,
            Gamma1_t,
            zeta1_t,
            alpha2_t,
            beta2_t,
            gamma2_t,
            zeta2_t,
            kappa_t,
        ) = (
            params["p_M"],
            params["p_A"],
            params["sigma_m"],
            params["sigma_r"],
            params["alpha1"],
            params["beta1"],
            params["Gamma1"],
            params["zeta1"],
            params["alpha2"],
            params["beta2"],
            params["gamma2"],
            params["zeta2"],
            params["kappa"],
        )
        At0 = np.repeat(0, n)
        At1 = np.repeat(1, n)
        epsilon_t = np.random.normal(0, sigma_m, size=[n, d])
        if t == 0:
            Mt1 = (
                np.linalg.solve(np.identity(d) - W0, epsilon_t.T).T
                + alpha1_t
                + At1[:, np.newaxis] @ beta1_t[np.newaxis, :]
            )
            Mt0 = (
                np.linalg.solve(np.identity(d) - W0, epsilon_t.T).T
                + alpha1_t
                + At0[:, np.newaxis] @ beta1_t[np.newaxis, :]
            )

            mu1 = alpha1_t + At1[:, np.newaxis] @ beta1_t[np.newaxis, :]
            Cov1 = Cov_Mt
            Mt1_Mj = sample_from_conditional_mvn(
                cond_idx=j, cond_val=c, mus=mu1, Cov=Cov1
            )

            mu0 = alpha1_t + At0[:, np.newaxis] @ beta1_t[np.newaxis, :]
            Cov0 = Cov_Mt
            Mt0_Mj = sample_from_conditional_mvn(
                cond_idx=j, cond_val=c, mus=mu0, Cov=Cov0
            )

            mu_R1 = alpha2_t + beta2_t * At1 + (Mt1 @ kappa_t).squeeze()
            mu_R0 = alpha2_t + beta2_t * At0 + (Mt0 @ kappa_t).squeeze()

            mu_R1_Mj = alpha2_t + beta2_t * At1 + (Mt1_Mj @ kappa_t).squeeze()
            mu_R0_Mj = alpha2_t + beta2_t * At0 + (Mt0_Mj @ kappa_t).squeeze()

            Rt1 = np.random.normal(loc=mu_R1, scale=sigma_r)
            Rt0 = np.random.normal(loc=mu_R0, scale=sigma_r)

            Rt1_Mj = np.random.normal(loc=mu_R1_Mj, scale=sigma_r)
            Rt0_Mj = np.random.normal(loc=mu_R0_Mj, scale=sigma_r)

        else:
            Mt1 = (
                np.linalg.solve(np.identity(d) - W0, epsilon_t.T).T
                + alpha1_t
                + (Gamma1_t @ Mt1_1.T).T
                + At1[:, np.newaxis] @ beta1_t[np.newaxis, :]
                + Rt1_1[:, np.newaxis] @ zeta1_t[np.newaxis, :]
            )
            Mt0 = (
                np.linalg.solve(np.identity(d) - W0, epsilon_t.T).T
                + alpha1_t
                + (Gamma1_t @ Mt0_1.T).T
                + At0[:, np.newaxis] @ beta1_t[np.newaxis, :]
                + Rt0_1[:, np.newaxis] @ zeta1_t[np.newaxis, :]
            )

            mu1 = (
                alpha1_t
                + (Gamma1_t @ Mt1_1_Mj.T).T
                + At1[:, np.newaxis] @ beta1_t[np.newaxis, :]
                + Rt1_1_Mj[:, np.newaxis] @ zeta1_t[np.newaxis, :]
            )
            Cov1 = Cov_Mt
            Mt1_Mj = sample_from_conditional_mvn(
                cond_idx=j, cond_val=c, mus=mu1, Cov=Cov1
            )

            mu0 = (
                alpha1_t
                + (Gamma1_t @ Mt0_1_Mj.T).T
                + At0[:, np.newaxis] @ beta1_t[np.newaxis, :]
                + Rt0_1_Mj[:, np.newaxis] @ zeta1_t[np.newaxis, :]
            )
            Cov0 = Cov_Mt
            Mt0_Mj = sample_from_conditional_mvn(
                cond_idx=j, cond_val=c, mus=mu0, Cov=Cov0
            )

            mu_R1 = (
                alpha2_t
                + beta2_t * At1
                + zeta2_t * Rt1_1
                + (Mt1_1 @ gamma2_t).squeeze()
                + (Mt1 @ kappa_t).squeeze()
            )
            mu_R0 = (
                alpha2_t
                + beta2_t * At0
                + zeta2_t * Rt0_1
                + (Mt0_1 @ gamma2_t).squeeze()
                + (Mt0 @ kappa_t).squeeze()
            )

            mu_R1_Mj = (
                alpha2_t
                + beta2_t * At1
                + zeta2_t * Rt1_1_Mj
                + (Mt1_1_Mj @ gamma2_t).squeeze()
                + (Mt1_Mj @ kappa_t).squeeze()
            )
            mu_R0_Mj = (
                alpha2_t
                + beta2_t * At0
                + zeta2_t * Rt0_1_Mj
                + (Mt0_1_Mj @ gamma2_t).squeeze()
                + (Mt0_Mj @ kappa_t).squeeze()
            )

            Rt1 = np.random.normal(loc=mu_R1, scale=sigma_r)
            Rt0 = np.random.normal(loc=mu_R0, scale=sigma_r)

            Rt1_Mj = np.random.normal(loc=mu_R1_Mj, scale=sigma_r)
            Rt0_Mj = np.random.normal(loc=mu_R0_Mj, scale=sigma_r)

        Mt1_1 = copy.deepcopy(Mt1)
        Mt0_1 = copy.deepcopy(Mt0)
        Rt1_1 = copy.deepcopy(Rt1)
        Rt0_1 = copy.deepcopy(Rt0)
        Mt1_1_Mj = copy.deepcopy(Mt1_Mj)
        Mt0_1_Mj = copy.deepcopy(Mt0_Mj)
        Rt1_1_Mj = copy.deepcopy(Rt1_Mj)
        Rt0_1_Mj = copy.deepcopy(Rt0_Mj)

        Rt_delta[t] = np.mean(Rt1 - Rt0)
        Rt_Mj_delta[t] = np.mean(Rt1_Mj - Rt0_Mj)
        etajs[t] = np.mean(Rt_delta[: (t + 1)]) - np.mean(Rt_Mj_delta[: (t + 1)])
    # times t
    etajs = etajs * np.arange(1, T + 1)
    return etajs


if __name__ == "__main__":
    pass
