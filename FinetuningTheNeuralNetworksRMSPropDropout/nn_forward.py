import numpy as np
from function import softmax, sigmoid
from NN import apply_dropout_forward  # ⭐ 正确引入 Dropout Forward

def nn_forward(nn, batch_x, batch_y):
    s = len(nn.cost) + 1
    batch_x = batch_x.T
    batch_y = batch_y.T
    m = batch_x.shape[1]
    nn.a[0] = batch_x
    is_training = (batch_y is not None)

    cost2 = 0
    for k in range(1, nn.depth):
        z = np.dot(nn.W[k-1], nn.a[k-1]) + np.tile(nn.b[k-1], (1, m))

        # ----------- Batch Normalization -----------
        if nn.batch_normalization:
            nn.E[k-1] = nn.E[k-1] * nn.vecNum + np.array([np.sum(z, axis=1)]).T
            nn.S[k-1] = nn.S[k-1]**2 * (nn.vecNum - 1) + np.array([(m - 1) *
                             np.std(z, ddof=1, axis=1)**2]).T
            nn.vecNum += m
            nn.E[k-1] /= nn.vecNum
            nn.S[k-1] = np.sqrt(nn.S[k-1] / (nn.vecNum - 1))
            z = (z - np.tile(nn.E[k-1], (1, m))) / \
                np.tile(nn.S[k-1] + 1e-4*np.ones(nn.S[k-1].shape), (1, m))
            z = nn.Gamma[k-1] * z + nn.Beta[k-1]

        # ----------- Activation ----------
        if k == nn.depth - 1:
            f = nn.output_function
        else:
            f = nn.active_function

        if f == 'sigmoid':
            a = sigmoid(z)
        elif f == 'tanh':
            a = np.tanh(z)
        elif f == 'relu':
            a = np.maximum(z, 0)
        elif f == 'softmax':
            a = softmax(z)

        # ----------- ⭐ Dropout (Hidden Layers Only)-----------
        if f != 'softmax':  # ⭐避免对输出层使用Dropout
            a = apply_dropout_forward(nn, k, a, is_training=is_training)

        nn.a[k] = a
        cost2 += np.sum(nn.W[k-1] ** 2)

    # ----------- Cost Computation -----------
    eps = 1e-10
    if nn.objective_function == 'MSE':
        nn.cost[s] = 0.5 / m * np.sum((nn.a[k] - batch_y)**2) + 0.5 * nn.weight_decay * cost2
    elif nn.objective_function == 'Cross Entropy':
        nn.cost[s] = -np.sum(batch_y * np.log(nn.a[k] + eps)) / m + \
                      0.5 * nn.weight_decay * cost2

    return nn
