# PyTorch (DeepSVDD) and TensorFlow (the surrogate network) run in the same
# process. If TensorFlow is imported first, torch's lazy import of
# torch._dynamo (on the first torch optimizer) aborts the process with
# "free(): invalid pointer". Importing it before anything here imports
# TensorFlow avoids that.
try:
    import torch._dynamo  # noqa: F401
except ImportError:  # torch is only needed for DeepSVDD
    pass
