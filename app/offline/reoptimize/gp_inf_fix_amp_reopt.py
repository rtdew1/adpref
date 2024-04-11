import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel


def K(inputs1, inputs2, hypers):
    amp = hypers["amp"]
    ls = hypers["ls"]
    if len(inputs1.shape) == 1:
        inputs1 = inputs1.reshape(-1, 1)
    if len(inputs2.shape) == 1:
        inputs2 = inputs2.reshape(-1, 1)
    N1 = inputs1.shape[0]
    N2 = inputs2.shape[0]
    out = amp ** 2 * rbf_kernel(inputs1, inputs2, 1 / (2 * ls ** 2))
    if N1 == N2:
        if (inputs1 == inputs2).all():
            out += np.diag(1e-6 * np.ones(N1))

    return out


def K_self_diag(x, hypers):
    """
    Efficiently compute np.diagonal(K(x, x, hypers)), which is just `amp ** 2`
    """

    return np.full(shape=(x.shape[0]), fill_value=hypers["amp"] ** 2)

def diag_dot_product(x, y):
    """
    Efficiently compute np.diagonal(x @ y)

    Reference: https://stackoverflow.com/a/14759341
    """

    return np.einsum('ij,ji->i', x, y)


def compute_gp(new_x, x, y, hypers, compute_cov=True):
    noise = hypers["noise"]

    K_newx_x = K(new_x, x, hypers)

    K_sigma = K(x, x, hypers) + np.diag((noise ** 2) * np.ones(x.shape[0]))
    K_coef =  K_newx_x @ np.linalg.inv(K_sigma)
    mean = K_coef @ y

    if compute_cov:
        var = K_self_diag(new_x, hypers) - diag_dot_product(K_coef, K_newx_x.T)
        return (mean, var)
    else:
        return mean


def nlml(pars, data):
    x = data["x"]
    y = data["y"]
    N = x.shape[0]
    noise = pars[0]
    hypers = {"amp": 1, "ls": pars[1]}
    K_y = K(x, x, hypers) + np.diag(noise ** 2 * np.ones(N))
    _, logdet_K_y = np.linalg.slogdet(K_y)
    return 0.5 * y.T @ np.linalg.inv(K_y) @ y + 0.5 * logdet_K_y + 0.5 * N * np.log(2 * np.pi)


def optim_hypers(pars0, data):
    bnds = ((1e-6, 100), (1e-6, 100))
    opt_out = minimize(nlml, pars0, args=data, method="L-BFGS-B", bounds=bnds)
    return {"noise": opt_out.x[0], "amp": 1, "ls": opt_out.x[1]}


def nlmap(pars, args):
    x = args["x"]
    y = args["y"]
    N = x.shape[0]

    noise = pars[0]
    amp = 1
    ls = pars[1]

    hypers = {"amp": amp, "ls": ls}
    K_y = K(x, x, hypers) + np.diag(noise ** 2 * np.ones(N))

    # log-marginal likelihood
    _, logdet_K_y = np.linalg.slogdet(K_y)
    lml = -0.5 * y.T @ np.linalg.inv(K_y) @ y - 0.5 * logdet_K_y - 0.5 * N * np.log(2 * np.pi)

    # log prior for noise: default is noise ~ gamma(5,0.05)
    if "noise_prior" in args.keys():
        lp_noise = stats.gamma.logpdf(noise, a=args["noise_prior"]["a"], scale=args["noise_prior"]["scale"])
    else:
        lp_noise = stats.gamma.logpdf(noise, a=5, scale=0.05)

    # log prior for lengthscale: default is lengthscale ~ inv-gamma(14,14)
    if "ls_prior" in args.keys():
        lp_ls = stats.invgamma.logpdf(ls, a=args["ls_prior"]["a"], scale=args["ls_prior"]["scale"])
    else:
        lp_ls = stats.invgamma.logpdf(ls, a=14, scale=14)

    return -(lml + lp_noise + lp_ls)


def map_hypers(pars0, args, fixed_noise=None):
    bnds = ((1e-6, 100), (1e-6, 100))
    opt_out = minimize(nlmap, pars0, args=args, method="L-BFGS-B", bounds=bnds)

    return {"noise": opt_out.x[0], "amp": 1, "ls": opt_out.x[1]}
