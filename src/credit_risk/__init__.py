"""Credit risk demo package."""
__all__ = ["train_demo", "score_application"]

def train_demo(*args, **kwargs):
    from .pipeline import train_demo as _train_demo
    return _train_demo(*args, **kwargs)

def score_application(*args, **kwargs):
    from .pipeline import score_application as _score_application
    return _score_application(*args, **kwargs)
