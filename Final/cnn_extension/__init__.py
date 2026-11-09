"""
CNN extension of the GECCO 2026 method
"Evolving Multi-Channel Confidence-Aware Activation Functions for Missing Data
with Channel Propagation".

The original method (GP representation, protected operators, fitness, fully
connected ChannelProp, 3C-MLP) lives unchanged in ``gp_core``.  This package
adds convolutional ChannelProp, three-stream pooling, image backbones,
spatial-missingness corruption, train-only preprocessing, a backbone-agnostic
GP evaluator and a reproducible experiment runner.  See README.md.
"""
