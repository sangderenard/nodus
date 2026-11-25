"""Training loss of the two-layer tensor network row.

The forward is the one in tensor_network_row.py; the loss is the mean squared
error that train_aapl_qualified_row.PolicyNetwork.step minimises.
tools/compile_tensor_network_train_row.py records this function on an
SSATensorProgram, and Turing's compile_native_training_schedule derives the
backward pass and the native SGD update from it. No gradient is written here.
"""


def tensor_network_train_loss(features, targets, input_weight, hidden_bias, output_weight, output_bias):
    hidden = (features.matmul(input_weight) + hidden_bias).tanh()
    prediction = (hidden.matmul(output_weight) + output_bias).tanh()
    error = prediction - targets
    return (error * error).mean(), prediction
