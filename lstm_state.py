import numpy as np


class lstm_state:
    def __init__(self, units, x_dim, series):
        self.cell_temp_values = np.zeros((units, series))
        self.input_values = np.zeros((units, series))
        self.forget_values = np.zeros((units, series))
        self.output_values = np.zeros((units, series))
        self.cell_values = np.zeros((units, series))
        self.h = np.zeros((units, series))

        self.diff_h_values = np.zeros_like(self.h)
        self.diff_cell_values = np.zeros_like(self.cell_values)

    def clean_grads(self):
        self.diff_h_values = np.zeros_like(self.h)
        self.diff_cell_values = np.zeros_like(self.cell_values)
