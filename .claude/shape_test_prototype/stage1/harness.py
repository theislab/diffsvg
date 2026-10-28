"""Verification harness: seed the posterior-predictive draws (unseeded in diffsvg) so runs are comparable."""
import functools
import pymc as pm

_orig = pm.sample_posterior_predictive


@functools.wraps(_orig)
def _seeded(*args, **kwargs):
    kwargs.setdefault("random_seed", 7)
    return _orig(*args, **kwargs)


pm.sample_posterior_predictive = _seeded
