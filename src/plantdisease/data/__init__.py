"""Data loading and splitting utilities for plant disease classification.

Names are imported lazily, so lightweight users (the Gradio app only needs
`plantdisease.data.transforms`) do not have to install the `datasets` library.
"""
import importlib

_EXPORTS = {
    "load_plant_dataset": "dataset",
    "load_and_split": "splits",
    "HFImageDataset": "loaders",
    "make_loaders": "loaders",
    "build_eval_transform": "transforms",
    "build_train_transform": "transforms",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
