"""Namespace for high-level data loader functionality for lazy loading."""

from erlab.io.dataloader import loaders

load = loaders.load
loader_context = loaders.loader_context
set_data_dir = loaders.set_data_dir
set_loader = loaders.set_loader
extend_loader = loaders.extend_loader
summarize = loaders.summarize
