"""Batched two-layer inference row; weights are Nodus tensor inputs."""


def tensor_network_row(features, input_weight, hidden_bias, output_weight, output_bias):
    hidden = (features @ input_weight + hidden_bias).tanh()
    return (hidden @ output_weight + output_bias).tanh()
