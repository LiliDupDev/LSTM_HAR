from lstm_params import lstm_param
from lstm_state import lstm_state
import numpy as np
import utils as util
from sklearn import metrics


class lstm_model:

    def __init__(self, features, target, classes, units, batch, learning_rate ,beta_1, beta_2):
        self.X = features
        self.Y = target
        self.classes = classes
        self.units = units
        self.batch_size = batch
        self.learning_rate = learning_rate
        self.loss_history=list()

        z, x, y = features.shape

        self.x_dim = x
        self.h_dim = x + units
        self.params = lstm_param(units, x, y, classes, beta_1, beta_2)

    def train(self, epochs,verbose=True):
        samples, features, series = self.X.shape

        for epoch in range(epochs):

            # create batches
            batch_set = util.get_mini_batches(self.X, self.Y, self.batch_size)
            cell_prev = None
            h_prev = None
            cell_next = None
            h_next = None
            acc_grad = np.zeros((2, 9))  # Accumulated gradient

            # LSTM recurrent
            state = lstm_state(self.units, self.x_dim, series)

            # Iterations over batchs
            loss_batch=0

            y_true = np.zeros((self.Y.shape[0],1))
            y_predicted = np.zeros((self.Y.shape[0],1))
            cont_total = 0

            for current_batch in batch_set:
                x_current = current_batch[0]
                y_current = current_batch[1]
                samples = x_current.shape[0]

                # Cell state
                if cell_prev is None: cell_prev = np.zeros_like(state.cell_values)
                if h_prev is None: h_prev = np.zeros_like(state.h)

                loss = 0

                # Iterations by value
                for step in range(samples):
                    time_serie = x_current[step]
                    target = np.asmatrix(y_current[step])

                    xc = np.vstack((time_serie, h_prev))

                    # Forward
                    predicted, step_loss, state, grad_i, encoded_result, label = self.forward_step(xc, target, h_prev, cell_prev, state)
                    loss += step_loss

                    self.back_step(xc, grad_i, state, h_prev, cell_prev)

                    h_prev = state.h
                    cell_prev = state.cell_values

                    # to measure acc
                    y_true[cont_total] = np.argmax(target,axis=1)
                    y_predicted[cont_total] = label
                    cont_total += 1



                loss_batch+=loss/samples
                self.params.apply_diff(self.learning_rate, samples)

            epoch_loss=loss_batch/len(batch_set)
            self.loss_history.append(epoch_loss)
            if verbose:
                print("Epoch:"+str(epoch) + "  Loss: " +str(epoch_loss) + " " + self.measure_accuracy(y_true,y_predicted))

        return y_true,y_predicted

    # x = time series matrix
    # h_prev = hidden
    def forward_step(self, xc, target, h_prev, cell_prev, state):
        prob = 1

        features = 9  # TODO: Cambiar por una variable

        matrix_target = np.hstack([target.T] * features)

        # LSTM

        state.cell_temp_values = np.tanh(np.dot(self.params.wg, xc) + self.params.bg)
        state.input_values = util.sigmoid(np.dot(self.params.wi, xc) + self.params.bi)
        state.forget_values = util.sigmoid(np.dot(self.params.wf, xc) + self.params.bf)
        state.output_values = util.sigmoid(np.dot(self.params.wo, xc) + self.params.bo)
        state.cell_values = state.cell_temp_values * state.input_values + h_prev * state.forget_values
        state.h = state.cell_values * np.tanh(state.output_values)

        # Relu
        relu_layer = util.relu(np.dot(state.h, self.params.wr)+self.params.br)

        # Classification
        soft_arg = np.dot(self.params.wk.T, relu_layer) + self.params.bk
        # soft_arg = np.dot(self.params.wk.T, state.h) + self.params.bk
        predicted = util.softmax(soft_arg)
        softmax_result = np.argmax(predicted, axis=0)

        # print("sum features:", np.argmax(np.sum(predicted,axis=1)))
        label=np.argmax(np.sum(predicted,axis=1))
        prob = np.multiply(prob, np.sum(np.multiply(target.T, predicted), axis=1).reshape(-1, 1))
        result = np.sum(prob, axis=1, dtype="float32").T

        # Cross entropy loss
        loss = (-1/features)*(np.sum((np.dot(target,np.log(predicted)) + np.dot(1-target,np.log(1-predicted))),axis=1).reshape(-1,1))
        #loss = -np.log(np.dot(matrix_target, softmax_result))  # [1,2]

        grad = predicted - matrix_target

        return predicted, loss, state, grad, result, label



    def back_step(self, xc, grad, state, h_prev, cell_prev):
        # softmax
        self.params.wk_diff = np.dot(state.h, grad.T)
        # self.params.wk_diff = np.dot(grad,self.params.wr)
        self.params.bk_diff = grad


        # Relu
        dr_before = util.relu_derivative(np.dot(self.params.wk, grad))
        self.params.wr_diff += np.dot(dr_before.T, np.dot(state.h, self.params.wr)+self.params.br)
        self.params.br_diff += dr_before

        # h next state
        state.diff_h_values = np.dot(dr_before,self.params.wr)
        state.diff_h_values += state.h

        # output
        do = np.multiply(state.diff_h_values, util.tanh_normal(state.cell_values))
        db_o = np.multiply(do, np.multiply(state.output_values, (1 - state.output_values)))

        self.params.wo_diff += np.dot(db_o, xc.T)
        self.params.bo_diff += db_o

        # cell
        dc = np.multiply(state.diff_h_values,
                         np.multiply(state.output_values, (1 - util.tanh_normal(state.cell_values) ** 2)))
        dc += state.cell_values
        dc_temp = np.multiply(dc, state.input_values)
        db_c = np.multiply(dc_temp, (1 - state.cell_temp_values ** 2))

        self.params.wg_diff += np.dot(db_c, xc.T)
        self.params.bg_diff += db_c

        # input
        di = np.multiply(dc, state.cell_temp_values)
        db_i = np.multiply(di, np.multiply(state.input_values, (1 - state.input_values)))
        self.params.wi_diff += np.dot(db_i, xc.T)
        self.params.bi_diff += db_i

        # forget
        df = np.multiply(dc, cell_prev)
        db_f = np.multiply(df, np.multiply(state.forget_values, (1 - state.forget_values)))
        self.params.wf_diff += np.dot(db_f, xc.T)
        self.params.bf_diff += db_f

        # xc
        dxc = (np.dot(self.params.wf.T, db_f) +
               np.dot(self.params.wi.T, db_i) +
               np.dot(self.params.wg.T, db_c) +
               np.dot(self.params.wo.T, db_o)
               )

    def measure_accuracy(self, y, y_pred):
        # diff = np.sum(y - y_pred)
        # if diff < 0:
        #     diff = np.sum(y_pred-y)
        # n = y.shape[0]
        #acc = "Diff = "+str(diff)+" acc = "+ str(((n - diff) / n) * 100)
        m=metrics.accuracy_score(y, y_pred)
        acc =" acc = " + str(m)
        return acc

    def predict(self,x,y):
        samples = x.shape[0]
        y_predict = np.zeros((samples,1))
        h=None
        for s in range(samples):
            serie = x[s]
            if h is None: h = np.zeros((self.units, serie.shape[1]))
            y_predict[s], h = self.predict_series(serie,h)

        y_true = np.asmatrix(np.argmax(y,axis=1))
        print(self.measure_accuracy(y_true.T,y_predict))

    def predict_series(self, serie, h):
        xc = np.vstack((serie, h))

        cell_temp_values = np.tanh(np.dot(self.params.wg, xc) + self.params.bg)
        input_values = util.sigmoid(np.dot(self.params.wi, xc) + self.params.bi)
        forget_values = util.sigmoid(np.dot(self.params.wf, xc) + self.params.bf)
        output_values = util.sigmoid(np.dot(self.params.wo, xc) + self.params.bo)
        cell_values = cell_temp_values * (input_values + h) * forget_values
        h = cell_values * output_values

        # Relu
        relu_layer = util.relu(np.dot(h, self.params.wr) + self.params.br)

        # Classification
        soft_arg = np.dot(self.params.wk.T, relu_layer) + self.params.bk
        predicted = util.softmax(soft_arg)
        softmax_result = np.argmax(predicted, axis=0)

        label = np.argmax(np.sum(predicted, axis=1))

        return label, h
