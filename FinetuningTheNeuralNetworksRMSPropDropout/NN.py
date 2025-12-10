import numpy as np


class NN:
    def __init__(self, **arg):
        init = {'layer': [],
                'active_function': 'relu',
                'output_function': 'sigmoid',
                'learning_rate': 0.001,
                'weight_decay': 0,
                'cost': {},
                'batch_normalization': 0,
                'optimization_method': 'RMSProp_Nesterov',
                'objective_function': 'Cross Entropy',
                'dropout': 0.3  # ⭐默认Dropout 30%
                }

        param = dict()
        param.update(init)
        param.update(arg)

        self.batch_size = param['batch_size']
        self.size = param['layer']
        self.depth = len(self.size)
        self.active_function = param['active_function']
        self.output_function = param['output_function']
        self.learning_rate = param['learning_rate']
        self.weight_decay = param['weight_decay']
        self.cost = param['cost']
        self.batch_normalization = param['batch_normalization']
        self.optimization_method = param['optimization_method']
        self.objective_function = param['objective_function']
        self.dropout = param['dropout']
        self.a = dict()
        self.dropout_mask = dict()  # ⭐ Dropout mask
        self.vecNum = 0

        if self.optimization_method == 'Adam':
            self.AdamTime = 0

        if self.objective_function == 'Cross Entropy':
            self.output_function = 'softmax'

        # Parameters
        self.W = {};
        self.b = {}
        # Buffers
        self.vW = {};
        self.vb = {}
        self.rW = {};
        self.rb = {}
        self.sW = {};
        self.sb = {}

        # For BN
        self.E = {};
        self.S = {};
        self.Gamma = {};
        self.Beta = {}
        self.vGamma = {};
        self.rGamma = {};
        self.vBeta = {};
        self.rBeta = {}
        self.sGamma = {};
        self.sBeta = {}
        # Gradients
        self.W_grad = {};
        self.b_grad = {};
        self.delta = {}
        self.Gamma_grad = {};
        self.Beta_grad = {}

        for k in range(self.depth - 1):
            width = self.size[k]
            height = self.size[k + 1]

            # ⭐He init for ReLU
            self.W[k] = np.random.randn(height, width) * np.sqrt(2.0 / width)
            self.b[k] = np.zeros((height, 1))

            method = self.optimization_method
            if method == 'Momentum':
                self.vW[k] = np.zeros((height,width))
                self.vb[k] = np.zeros((height,1))

            if method == 'RMSProp_Nesterov':
                self.vW[k] = np.zeros((height,width))
                self.vb[k] = np.zeros((height,1))
                self.rW[k] = np.zeros((height,width))
                self.rb[k] = np.zeros((height,1))

            if method in ['AdaGrad','RMSProp','Adam']:
                self.rW[k] = np.zeros((height,width))
                self.rb[k] = np.zeros((height,1))

            if method == 'Adam':
                self.sW[k] = np.zeros((height,width))
                self.sb[k] = np.zeros((height,1))

            if self.batch_normalization:
                self.E[k] = np.zeros((height,1))
                self.S[k] = np.zeros((height,1))
                self.Gamma[k] = 1
                self.Beta[k] = 0
                self.vecNum = 0

                if method == 'Momentum':
                    self.vGamma[k] = 1
                    self.vBeta[k] = 0

                if method in ['AdaGrad','RMSProp','Adam']:
                    self.rGamma[k] = 0
                    self.rBeta[k] = 0

                if method == 'Adam':
                    self.sGamma[k] = 1
                    self.sBeta[k] = 0


# ==========================================================
# ⭐ Dropout Forward（训练阶段生效，测试自动关闭）
# ==========================================================
def apply_dropout_forward(nn, layer_idx, z, is_training=True):
    if nn.dropout > 0 and is_training and layer_idx < nn.depth - 2:
        keep_prob = 1 - nn.dropout
        mask = (np.random.rand(*z.shape) < keep_prob).astype(float)
        nn.dropout_mask[layer_idx] = mask
        return z * mask / keep_prob
    else:
        nn.dropout_mask[layer_idx] = np.ones_like(z)
        return z


# ==========================================================
# ⭐ Dropout Backward
# ==========================================================
def apply_dropout_backward(nn, layer_idx, dz):
    if nn.dropout > 0 and layer_idx < nn.depth - 2:
        mask = nn.dropout_mask[layer_idx]
        keep_prob = 1 - nn.dropout
        return dz * mask / keep_prob
    return dz


# ==========================================================
# ☆ Real RMSProp + Nesterov Momentum UPDATE（Algorithm 8.6）
# ==========================================================
def real_RMSProp_Nesterov_update(nn):
    eps = 1e-8
    rho = 0.9
    alpha = 0.8

    for k in range(nn.depth-1):
        gW = nn.W_grad[k]
        gb = nn.b_grad[k]

        nn.rW[k] = rho * nn.rW[k] + (1 - rho) * (gW * gW)
        nn.rb[k] = rho * nn.rb[k] + (1 - rho) * (gb * gb)

        vW_prev = nn.vW[k]
        vb_prev = nn.vb[k]

        nn.vW[k] = alpha * nn.vW[k] - nn.learning_rate * gW / (np.sqrt(nn.rW[k]) + eps)
        nn.vb[k] = alpha * nn.vb[k] - nn.learning_rate * gb / (np.sqrt(nn.rb[k]) + eps)

        nn.W[k] += -alpha * vW_prev + (1 + alpha) * nn.vW[k]
        nn.b[k] += -alpha * vb_prev + (1 + alpha) * nn.vb[k]

    return nn
