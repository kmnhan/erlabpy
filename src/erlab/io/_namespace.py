"""Namespace for high-level data loader functionality for lazy loading."""

from erlab.io.dataloader import _loaders

load = _loaders.load
loader_context = _loaders.loader_context
set_data_dir = _loaders.set_data_dir
set_loader = _loaders.set_loader
summarize = _loaders.summarize
