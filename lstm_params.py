# Class params will contain all parameters of the network
import numpy as np
import utils as util


class lstm_param:

    def __init__(self, units, x_dim, features, classes, beta_1, beta_2):
        self.units = units
        self.x_dim = x_dim
        concat_len = x_dim + units

        # weight matrices
        self.wg = util.rand_arr(-0.1, 0.1, units, concat_len)
        self.wi = util.rand_arr(-0.1, 0.1, units, concat_len)
        self.wf = util.rand_arr(-0.1, 0.1, units, concat_len)
        self.wo = util.rand_arr(-0.1, 0.1, units, concat_len)
        self.wk = util.rand_arr(-0.1, 0.1, units, classes)
        self.wr = util.rand_arr(-0.1, 0.1, features, features) # c1

        # bias terms
        self.bg = util.rand_arr(-0.1, 0.1, units, features)
        self.bi = util.rand_arr(-0.1, 0.1, units, features)
        self.bf = util.rand_arr(-0.1, 0.1, units, features)
        self.bo = util.rand_arr(-0.1, 0.1, units, features)
        self.bk = util.rand_arr(-0.1, 0.1, classes, features)
        self.br = util.rand_arr(-0.1, 0.1, units, features)# c1

        # diffs
        self.wg_diff = np.zeros((units, concat_len))
        self.wi_diff = np.zeros((units, concat_len))
        self.wf_diff = np.zeros((units, concat_len))
        self.wo_diff = np.zeros((units, concat_len))
        self.wk_diff = np.zeros((units, classes))
        self.wr_diff = np.zeros((features, features))# c1

        self.bg_diff = np.zeros((units, features))
        self.bi_diff = np.zeros((units, features))
        self.bf_diff = np.zeros((units, features))
        self.bo_diff = np.zeros((units, features))
        self.bk_diff = np.zeros((classes, features))
        self.br_diff = np.zeros((units, features))# c1

        # Adam's
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.adam_m_wg = np.zeros((units, concat_len))
        self.adam_m_wi = np.zeros((units, concat_len))
        self.adam_m_wf = np.zeros((units, concat_len))
        self.adam_m_wo = np.zeros((units, concat_len))
        self.adam_m_wk = np.zeros((units, classes))
        self.adam_m_wr = np.zeros((features, features))# c1

        self.adam_m_bg = np.zeros((units, features))
        self.adam_m_bi = np.zeros((units, features))
        self.adam_m_bf = np.zeros((units, features))
        self.adam_m_bo = np.zeros((units, features))
        self.adam_m_bk = np.zeros((classes, features))
        self.adam_m_br = np.zeros((units, features)) #c1

        self.adam_v_wg = np.zeros((units, concat_len))
        self.adam_v_wi = np.zeros((units, concat_len))
        self.adam_v_wf = np.zeros((units, concat_len))
        self.adam_v_wo = np.zeros((units, concat_len))
        self.adam_v_wk = np.zeros((units, classes))
        self.adam_v_wr = np.zeros((features, features)) #c1

        self.adam_v_bg = np.zeros((units, features))
        self.adam_v_bi = np.zeros((units, features))
        self.adam_v_bf = np.zeros((units, features))
        self.adam_v_bo = np.zeros((units, features))
        self.adam_v_bk = np.zeros((classes, features))
        self.adam_v_br = np.zeros((units, features)) #c1

    def apply_diff(self, lr=1, batch=1):
        # Cell hat gate
        self.adam_m_wg = self.adam_m_wg * self.beta_1 + (1 - self.beta_1) * self.wg_diff
        self.adam_v_wg = self.adam_v_wg * self.beta_2 + (1 - self.beta_2) * np.multiply(self.wg_diff, self.wg_diff)
        m_correlated = self.adam_m_wg / (1 - self.beta_1 ** batch)
        v_correlated = self.adam_v_wg / (1 - self.beta_2 ** batch)
        self.wg -= lr * m_correlated / (np.sqrt(v_correlated) + 1e-8)

        self.adam_m_bg = self.adam_m_bg * self.beta_1 + (1 - self.beta_1) * self.bg_diff
        self.adam_v_bg = self.adam_v_bg * self.beta_2 + (1 - self.beta_2) * np.multiply(self.bg_diff, self.bg_diff)
        m_correlated = self.adam_m_bg / (1 - self.beta_1 ** batch)
        v_correlated = self.adam_v_bg / (1 - self.beta_2 ** batch)
        self.bg -= lr * m_correlated / (np.sqrt(v_correlated) + 1e-8)

        # Input gate
        self.adam_m_wi = self.adam_m_wi * self.beta_1 + (1 - self.beta_1) * self.wi_diff
        self.adam_v_wi = self.adam_v_wi * self.beta_2 + (1 - self.beta_2) * np.multiply(self.wi_diff, self.wi_diff)
        m_correlated = self.adam_m_wi / (1 - self.beta_1 ** batch)
        v_correlated = self.adam_v_wi / (1 - self.beta_2 ** batch)
        self.wi -= lr * m_correlated / (np.sqrt(v_correlated) + 1e-8)

        self.adam_m_bi = self.adam_m_bi * self.beta_1 + (1 - self.beta_1) * self.bi_diff
        self.adam_v_bi = self.adam_v_bi * self.beta_2 + (1 - self.beta_2) * np.multiply(self.bi_diff, self.bi_diff)
        m_correlated = self.adam_m_bi / (1 - self.beta_1 ** batch)
        v_correlated = self.adam_v_bi / (1 - self.beta_2 ** batch)
        self.bi -= lr * m_correlated / (np.sqrt(v_correlated) + 1e-8)

        # Forget gate
        self.adam_m_wf = self.adam_m_wf * self.beta_1 + (1 - self.beta_1) * self.wf_diff
        self.adam_v_wf = self.adam_v_wf * self.beta_2 + (1 - self.beta_2) * np.multiply(self.wf_diff, self.wf_diff)
        m_correlated = self.adam_m_wf / (1 - self.beta_1 ** batch)
        v_correlated = self.adam_v_wf / (1 - self.beta_2 ** batch)
        self.wf -= lr * m_correlated / (np.sqrt(v_correlated) + 1e-8)

        self.adam_m_bf = self.adam_m_bf * self.beta_1 + (1 - self.beta_1) * self.bf_diff
        self.adam_v_bf = self.adam_v_bf * self.beta_2 + (1 - self.beta_2) * np.multiply(self.bf_diff, self.bf_diff)
        m_correlated = self.adam_m_bf / (1 - self.beta_1 ** batch)
        v_correlated = self.adam_v_bf / (1 - self.beta_2 ** batch)
        self.bf -= lr * m_correlated / (np.sqrt(v_correlated) + 1e-8)

        # Output gate
        self.adam_m_wo = self.adam_m_wo * self.beta_1 + (1 - self.beta_1) * self.wo_diff
        self.adam_v_wo = self.adam_v_wo * self.beta_2 + (1 - self.beta_2) * np.multiply(self.wo_diff, self.wo_diff)
        m_correlated = self.adam_m_wo / (1 - self.beta_1 ** batch)
        v_correlated = self.adam_v_wo / (1 - self.beta_2 ** batch)
        self.wo -= lr * m_correlated / (np.sqrt(v_correlated) + 1e-8)

        self.adam_m_bo = self.adam_m_bo * self.beta_1 + (1 - self.beta_1) * self.bo_diff
        self.adam_v_bo = self.adam_v_bo * self.beta_2 + (1 - self.beta_2) * np.multiply(self.bo_diff, self.bo_diff)
        m_correlated = self.adam_m_bo / (1 - self.beta_1 ** batch)
        v_correlated = self.adam_v_bo / (1 - self.beta_2 ** batch)
        self.bo -= lr * m_correlated / (np.sqrt(v_correlated) + 1e-8)

        # Softmax neuron
        self.adam_m_wk = self.adam_m_wk * self.beta_1 + (1 - self.beta_1) * self.wk_diff
        self.adam_v_wk = self.adam_v_wk * self.beta_2 + (1 - self.beta_2) * np.multiply(self.wk_diff, self.wk_diff)
        m_correlated = self.adam_m_wk / (1 - self.beta_1 ** batch)
        v_correlated = self.adam_v_wk / (1 - self.beta_2 ** batch)
        self.wk -= lr * m_correlated / (np.sqrt(v_correlated) + 1e-8)

        self.adam_m_bk = self.adam_m_bk * self.beta_1 + (1 - self.beta_1) * self.bk_diff
        self.adam_v_bk = self.adam_v_bk * self.beta_2 + (1 - self.beta_2) * np.multiply(self.bk_diff, self.bk_diff)
        m_correlated = self.adam_m_bk / (1 - self.beta_1 ** batch)
        v_correlated = self.adam_v_bk / (1 - self.beta_2 ** batch)
        self.bk -= lr * m_correlated / (np.sqrt(v_correlated) + 1e-8)

        # reset diffs to zero
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi)
        self.wf_diff = np.zeros_like(self.wf)
        self.wo_diff = np.zeros_like(self.wo)
        self.wk_diff = np.zeros_like(self.wk)

        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bf_diff = np.zeros_like(self.bf)
        self.bo_diff = np.zeros_like(self.bo)
        self.bk_diff = np.zeros_like(self.bk)

