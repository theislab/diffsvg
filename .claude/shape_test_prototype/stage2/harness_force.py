"""Pin every posterior-predictive draw to one seed, overriding whatever the caller passes (old-path comparison only)."""
import functools
import pymc as pm

_orig = pm.sample_posterior_predictive


@functools.wraps(_orig)
def _pinned(*args, **kwargs):
    kwargs["random_seed"] = 7
    return _orig(*args, **kwargs)


pm.sample_posterior_predictive = _pinned
