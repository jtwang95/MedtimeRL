import numpy as np
from dagma.linear import DagmaLinear

# from dagma.utils import *


def estimate_model_parameters(
    actions,
    mediators,
    rewards,
    estimate_W0=False,
    estimate_W0_lambda1=0.00,
    estimate_W0_threshold=0.3,
):
    n, T, d = mediators.shape

    intercept = np.ones([n, T - 1, 1])
    At = actions[:, 1:T, np.newaxis]
    Mt_1 = mediators[:, : (T - 1), :]
    Rt_1 = rewards[:, : (T - 1), np.newaxis]
    Mt = mediators[:, 1:T, :]
    Rt = rewards[:, 1:T]
    Xt = np.concatenate([intercept, At, Mt_1, Rt_1], axis=2).reshape([-1, d + 3])
    Zt = np.concatenate([intercept, At, Rt_1, Mt_1, Mt], axis=2).reshape(
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
        "Theta1": Theta1_hat,
        "Theta2": Theta2_hat,
    }

    if estimate_W0:
        model = DagmaLinear(loss_type="l2")
        W_est = model.fit(
            Mt_flat - Xt @ Theta1_hat,
            lambda1=estimate_W0_lambda1,
            w_threshold=estimate_W0_threshold,
        )
        params_est["W0"] = W_est.T

    # return alpha1_hat, beta1_hat, Gamma1_hat, zeta1_hat, alpha2_hat, beta2_hat, zeta2_hat, gamma2_hat, kappa_hat, Theta1_hat, Theta2_hat
    return params_est


if __name__ == "__main__":
    pass
