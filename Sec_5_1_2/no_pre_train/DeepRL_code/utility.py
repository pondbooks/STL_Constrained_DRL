import math
import torch

def calculate_log_pi(log_stds, noises, actions):
    """ Return the logarithmic probability density． """

    gaussian_log_probs = \
        (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    log_pis = gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

    return log_pis

def reparameterize(means, log_stds):
    """ Return a stochastic action and its probablistic density using Reparameterization Trick． """
    # standard deviation．
    stds = log_stds.exp()
    # Sampling a noise from a standard normal distribution.
    noises = torch.randn_like(means)
    # Compute the sample from N(means, stds) using Reparameterization Trick．
    us = means + noises * stds
    # Using tanh 
    actions = torch.tanh(us)

    # Compute logarithmic probability density for the action．
    log_pis = calculate_log_pi(log_stds, noises, actions)

    return actions, log_pis


