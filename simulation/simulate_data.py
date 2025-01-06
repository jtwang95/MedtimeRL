import numpy as np


def generate_model_parameters(n,
                              T,
                              d,
                              p_M,
                              p_A,
                              seed,
                              is_check_stationarity=True):
    np.random.seed(seed)
    sigma_m, sigma_r = 1, 1  # random noise for conditional models for M and R
    is_stationary = False
    max_attempts = 10
    attempt = 1
    #parameter values
    while not is_stationary and (attempt <= max_attempts):
        Theta1 = np.random.uniform(low=-0.4, high=0.5, size=[d + 3, d])
        alpha1 = Theta1[0, :]
        beta1 = Theta1[1, :]  # 1 * d
        Gamma1 = Theta1[2:(d + 2), :]  # d * d
        zeta1 = Theta1[d + 2, :]

        Theta2 = np.random.uniform(low=-0.5, high=0.6, size=[2 * d + 3])
        alpha2 = Theta2[0]
        beta2 = Theta2[1]
        zeta2 = Theta2[2]
        gamma2 = Theta2[3:(d + 3)]
        kappa = Theta2[(d + 3):(2 * d + 3)]

        E1 = np.tril(np.random.binomial(n=1, p=p_M, size=[d, d]))
        E2 = np.tril(
            np.random.uniform(low=0.5, high=0.9, size=[d, d]) *
            np.random.choice(
                a=[-1, 1], p=[0.5, 0.5], size=[d, d
                                               ], replace=True))  # may be DAG
        E1E2 = np.multiply(E1, E2)
        W0 = E1E2
        np.fill_diagonal(W0, 0)
        if is_check_stationarity:
            is_stationary = _check_stationarity_condition(kappa=kappa,
                                                          Gamma1=Gamma1,
                                                          zeta1=zeta1,
                                                          gamma2=gamma2,
                                                          zeta2=zeta2,
                                                          W0=W0)
        else:
            is_stationary = True
        if attempt == max_attempts:
            raise ValueError(
                "Stationarity condition check fails. Please adjust the range for each parameter."
            )
        attempt += 1

    return {
        "d": d,
        "p_M": p_M,
        "p_A": p_A,
        "T": T,
        "n": n,
        "sigma_m": sigma_m,
        "sigma_r": sigma_r,
        "alpha1": alpha1,
        "beta1": beta1,
        "Gamma1": Gamma1,
        "zeta1": zeta1,
        "alpha2": alpha2,
        "beta2": beta2,
        "gamma2": gamma2,
        "zeta2": zeta2,
        "kappa": kappa,
        "W0": W0,
        "Theta1": Theta1,
        "Theta2": Theta2,
        "seed": seed
    }


def simulate_dataset_infinite(params, nsim, warmup=5):
    n, T, d, p_M, p_A, sigma_m, sigma_r, alpha1, beta1, Gamma1, zeta1, alpha2, beta2, gamma2, zeta2, kappa, W0 = params[
        "n"], params["T"], params["d"], params["p_M"], params["p_A"], params[
            "sigma_m"], params["sigma_r"], params["alpha1"], params[
                "beta1"], params["Gamma1"], params["zeta1"], params[
                    "alpha2"], params["beta2"], params["gamma2"], params[
                        "zeta2"], params["kappa"], params["W0"]
    # print(W0)
    R_0 = np.repeat(0, n)  # reward at time 0
    M_0 = np.zeros([n, d])
    actions = np.zeros([nsim, n, T])
    mediators = np.zeros([nsim, n, T, d])
    rewards = np.zeros([nsim, n, T])
    mediator0 = np.zeros([nsim, n, 1, d])
    reward0 = np.zeros([nsim, n, 1])
    for nidx in range(nsim):
        Rt_1 = R_0
        Mt_1 = M_0
        for t in range(T + warmup):
            At = np.random.binomial(n=1, p=p_A, size=[n])
            epsilon_t = np.random.normal(loc=0, scale=sigma_m, size=[n, d])
            Mt = np.linalg.solve(
                np.identity(d) - W0,
                epsilon_t.T).T + alpha1 + (Gamma1 @ Mt_1.T).T + (
                    beta1[:, np.newaxis] @ At[np.newaxis, :]).T + (
                        zeta1[:, np.newaxis] @ Rt_1[np.newaxis, :]).T
            mu_r = alpha2 + beta2 * At + zeta2 * Rt_1 + (
                Mt_1 @ gamma2).squeeze() + (Mt @ kappa).squeeze()
            Rt = np.random.normal(loc=mu_r, scale=sigma_r, size=[n])
            if (t >= warmup):
                actions[nidx, :, t - warmup] = At
                mediators[nidx, :, t - warmup, :] = Mt
                rewards[nidx, :, t - warmup] = Rt
                if (t == warmup):
                    mediator0[nidx, :, 0, :] = Mt_1
                    reward0[nidx, :, 0] = Rt_1
            Rt_1 = Rt
            Mt_1 = Mt
    return mediators, actions, rewards


def simulate_dataset_finite_old(params, nsim=100):
    n, T, d, p_M, p_A, sigma_m, sigma_r, alpha1, beta1, Gamma1, zeta1, alpha2, beta2, gamma2, zeta2, kappa, W0 = params[
        "n"], params["T"], params["d"], params["p_M"], params["p_A"], params[
            "sigma_m"], params["sigma_r"], params["alpha1"], params[
                "beta1"], params["Gamma1"], params["zeta1"], params[
                    "alpha2"], params["beta2"], params["gamma2"], params[
                        "zeta2"], params["kappa"], params["W0"]
    actions = np.zeros([nsim, n, T])
    mediators = np.zeros([nsim, n, T, d])
    rewards = np.zeros([nsim, n, T])
    for nidx in range(nsim):
        for t in range(T):
            term1 = np.sin(t) / 4
            term2 = np.cos(t) / 4
            # term1 = 0
            # term2 = 0
            alpha1_t = alpha1 + term1
            beta1_t = beta1 + term1
            Gamma1_t = Gamma1 + term1
            zeta1_t = zeta1 + term1
            alpha2_t = alpha2 + term2
            beta2_t = beta2 + term2
            zeta2_t = zeta2 + term2
            gamma2_t = gamma2 + term2
            kappa_t = kappa + term2

            At = np.random.binomial(n=1, p=p_A, size=[n])
            epsilon_t = np.random.normal(loc=0, scale=sigma_m, size=[n, d])
            if t == 0:
                Mt = np.linalg.solve(
                    np.identity(d) - W0, epsilon_t.T).T + alpha1_t + (
                        beta1_t[:, np.newaxis] @ At[np.newaxis, :]).T
                mu_r = alpha2_t + beta2_t * At + (Mt @ kappa_t).squeeze()
                Rt = np.random.normal(loc=mu_r, scale=sigma_r, size=[n])
            else:
                Mt = np.linalg.solve(
                    np.identity(d) - W0,
                    epsilon_t.T).T + alpha1_t + (Gamma1_t @ Mt_1.T).T + (
                        beta1_t[:, np.newaxis] @ At[np.newaxis, :]).T + (
                            zeta1_t[:, np.newaxis] @ Rt_1[np.newaxis, :]).T
                mu_r = alpha2_t + beta2_t * At + zeta2_t * Rt_1 + (
                    Mt_1 @ gamma2_t).squeeze() + (Mt @ kappa_t).squeeze()
                Rt = np.random.normal(loc=mu_r, scale=sigma_r, size=[n])
            actions[nidx, :, t] = At
            mediators[nidx, :, t, :] = Mt
            rewards[nidx, :, t] = Rt
            Rt_1 = Rt
            Mt_1 = Mt
    return mediators, actions, rewards


def simulate_dataset_finite(n, T, d, p_M, p_A, seed, nsim=100):
    params_list = {}
    for t in range(T):
        params_list[t] = generate_model_parameters(n,
                                                   T,
                                                   d,
                                                   p_M,
                                                   p_A,
                                                   seed=seed + t)
        if t == 0:
            W0 = params_list[t]["W0"]
        else:
            params_list[t]["W0"] = W0

    actions = np.zeros([nsim, n, T])
    mediators = np.zeros([nsim, n, T, d])
    rewards = np.zeros([nsim, n, T])
    for nidx in range(nsim):
        for t in range(T):
            params = params_list[t]
            p_M, p_A, sigma_m, sigma_r, alpha1_t, beta1_t, Gamma1_t, zeta1_t, alpha2_t, beta2_t, gamma2_t, zeta2_t, kappa_t, W0 = params[
                "p_M"], params["p_A"], params["sigma_m"], params[
                    "sigma_r"], params["alpha1"], params["beta1"], params[
                        "Gamma1"], params["zeta1"], params["alpha2"], params[
                            "beta2"], params["gamma2"], params[
                                "zeta2"], params["kappa"], params["W0"]

            At = np.random.binomial(n=1, p=p_A, size=[n])
            epsilon_t = np.random.normal(loc=0, scale=sigma_m, size=[n, d])
            if t == 0:
                Mt = np.linalg.solve(
                    np.identity(d) - W0, epsilon_t.T).T + alpha1_t + (
                        beta1_t[:, np.newaxis] @ At[np.newaxis, :]).T
                mu_r = alpha2_t + beta2_t * At + (Mt @ kappa_t).squeeze()
                Rt = np.random.normal(loc=mu_r, scale=sigma_r, size=[n])
            else:
                Mt = np.linalg.solve(
                    np.identity(d) - W0,
                    epsilon_t.T).T + alpha1_t + (Gamma1_t @ Mt_1.T).T + (
                        beta1_t[:, np.newaxis] @ At[np.newaxis, :]).T + (
                            zeta1_t[:, np.newaxis] @ Rt_1[np.newaxis, :]).T
                mu_r = alpha2_t + beta2_t * At + zeta2_t * Rt_1 + (
                    Mt_1 @ gamma2_t).squeeze() + (Mt @ kappa_t).squeeze()
                Rt = np.random.normal(loc=mu_r, scale=sigma_r, size=[n])
            actions[nidx, :, t] = At
            mediators[nidx, :, t, :] = Mt
            rewards[nidx, :, t] = Rt
            Rt_1 = Rt
            Mt_1 = Mt
    return mediators, actions, rewards, params_list


def _check_stationarity_condition(kappa, Gamma1, zeta1, gamma2, zeta2, W0):
    d = Gamma1.shape[0]
    V = np.block([[Gamma1, zeta1.reshape(-1, 1)],
                  [gamma2.reshape(1, -1),
                   zeta2.reshape(1, 1)]])
    U = np.block([[np.zeros([d, d]), np.zeros([d, 1])],
                  [kappa.reshape(1, -1),
                   np.zeros([1, 1])]])

    # check stationarity condition for M and R
    M = np.linalg.solve(np.identity(d + 1) - U, V)
    w = np.linalg.eigvals(M)
    is_stationary_MR = (np.sum(np.abs(w) >= 1) == 0)

    # check stationarity condition for Stj
    T0 = 20
    T1 = 50

    is_finite_Sjt = True

    for j in range(d):
        B = np.zeros([T1, d])
        Cj = np.zeros([T1])
        Sj = np.zeros([T1 - 1])

        B1 = np.linalg.inv(np.identity(d) - W0)[:, j]
        C1j = np.sum(B1 * kappa)
        B[0, :] = B1.flatten()
        Cj[0] = C1j

        for t in range(1, T1):
            B[t] = (Gamma1 @ B[t - 1] + zeta1 * Cj[t - 1]).squeeze()
            Cj[t] = zeta2 * Cj[t - 1] + np.inner(B[t], kappa) + np.inner(
                B[t - 1], gamma2)
            if (t >= 2):
                Sj[t - 1] = Cj[t] - B[t, j] * Cj[0] - np.sum(
                    B[1:t, j] * Sj[(t - 2)::-1])
            else:
                Sj[t - 1] = Cj[t] - B[t, j] * Cj[0]
        ratioSj = np.abs(Sj[T1 - 2]) / np.abs(Sj[T0])
        if ratioSj < 1:
            pass
        else:
            is_finite_Sjt = False
            break
    # print("ratioSj:{}".format(ratioSj))
    # ------------ check whether Stj staisfies stationarity condition ------------ #
    # lag_max = 30
    # for j in range(d):
    #     Bts = np.zeros([lag_max])
    #     Bt_1 = np.linalg.inv(np.identity(d) - W0)[:, j]
    #     Cjt_1 = np.sum(Bt_1 * kappa)
    #     Bts[0] = 1.0  # M1j->M1j
    #     for i in range(1, lag_max):
    #         Bt = (Gamma1 @ Bt_1 + zeta1 * Cjt_1).squeeze()
    #         Cjt = zeta2 * Cjt_1 + np.sum(kappa * Bt) + np.sum(gamma2 * Bt_1)
    #         Bts[i] = Bt[j]
    #         Bt_1 = Bt
    #         Cjt_1 = Cjt
    #     A = np.array(Bts[1:(lag_max - 1)]).reshape(1, -1) * -1
    #     B = np.array([Bts[lag_max - 1]]).reshape(1, 1) * -1
    #     C = np.identity(lag_max - 2)
    #     D = np.zeros([lag_max - 2]).reshape(-1, 1)

    #     M = np.block([[A, B], [C, D]])
    #     w = np.linalg.eigvals(M)
    #     if np.sum(np.abs(w) >= 1) == 0:
    #         pass
    #     else:
    #         is_finite_Sjt = False
    #         break

    is_stationary = is_stationary_MR and is_finite_Sjt

    return is_stationary


if __name__ == "__main__":
    seed = 142020529
    d = 5
    params = generate_model_parameters(n=np.nan,
                                       T=np.nan,
                                       d=d,
                                       p_M=0.5,
                                       p_A=0.5,
                                       seed=seed)
