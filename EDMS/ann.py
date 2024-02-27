import numpy as np
from scipy.linalg import pinv
import copy

class Layer:
    '''Layer'''
    def __init__(self, w, b, neure_number, transfer_function, layer_index):
        self.transfer_function = transfer_function
        self.neure_number = neure_number
        self.layer_index = layer_index
        self.w = w
        self.b = b

class NetStruct:
    '''Structure'''
    def __init__(self, ni, nh, no, active_fun_list):
        self.neurals = []
        self.neurals.append(ni)
        if isinstance(nh, list):
            self.neurals.extend(nh)
        else:
            self.neurals.append(nh)
        self.neurals.append(no)
        if len(self.neurals)-2 == len(active_fun_list):
            active_fun_list.append('line')
        self.active_fun_list = active_fun_list
        self.layers = []
        for i in range(len(self.neurals)):
            if i == 0:
                self.layers.append(Layer([], [], self.neurals[i], 'none', i))
                continue
            f = self.neurals[i - 1]
            s = self.neurals[i]
            self.layers.append(Layer(np.random.randn(s, f), np.random.randn(s, 1), self.neurals[i], self.active_fun_list[i-1], i))

class NeuralNetwork:
    '''NN'''
    def __init__(self, net_struct, mu=1e-3, beta=10, iteration=100, tol=0.1):
        self.net_struct = net_struct
        self.mu = mu
        self.beta = beta
        self.iteration = iteration
        self.tol = tol

    def sim(self, x):
        self.net_struct.x = x
        self.forward()
        layer_num = len(self.net_struct.layers)
        predict = self.net_struct.layers[layer_num - 1].output_val
        return predict

    def actFun(self, z, active_type='sigm'):
        if active_type == 'sigm':
            f = 1.0 / (1.0 + np.exp(-z))
        elif active_type == 'tanh':
            f = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        elif active_type == 'radb':
            f = np.exp(-z * z)
        elif active_type == 'line':
            f = z
        return f

    def actFunGrad(self, z, active_type='sigm'):
        y = self.actFun(z, active_type)
        if active_type == 'sigm':
            grad = y * (1.0 - y)
        elif active_type == 'tanh':
            grad = 1.0 - y * y
        elif active_type == 'radb':
            grad = -2.0 * z * y
        elif active_type == 'line':
            grad = np.ones(z.shape)
        return grad

    def forward(self):
        layer_num = len(self.net_struct.layers)
        for i in range(layer_num):
            if i == 0:
                curr_layer = self.net_struct.layers[i]
                curr_layer.input_val = self.net_struct.x
                curr_layer.output_val = self.net_struct.x
                continue
            before_layer = self.net_struct.layers[i - 1]
            curr_layer = self.net_struct.layers[i]
            curr_layer.input_val = curr_layer.w.dot(before_layer.output_val) + curr_layer.b
            curr_layer.output_val = self.actFun(curr_layer.input_val, self.net_struct.active_fun_list[i - 1])

    def backward(self):
        layer_num = len(self.net_struct.layers)
        last_layer = self.net_struct.layers[layer_num - 1]
        last_layer.error = -self.actFunGrad(last_layer.input_val, self.net_struct.active_fun_list[layer_num - 2])
        for i in range(layer_num - 2, 0, -1):
            curr_layer = self.net_struct.layers[i]
            curr_layer.error = (last_layer.w.T.dot(last_layer.error)) * self.actFunGrad(curr_layer.input_val, self.net_struct.active_fun_list[i - 1])
            last_layer = curr_layer

    def parDeriv(self):
        layer_num = len(self.net_struct.layers)
        for i in range(1, layer_num):
            before_layer = self.net_struct.layers[i - 1]
            curr_layer = self.net_struct.layers[i]
            curr_error = curr_layer.error.reshape(curr_layer.error.size, 1, order='F')
            a = np.repeat(before_layer.output_val.T, curr_layer.neure_number, axis=0)
            tmp_w_par_deriv = curr_error * a
            curr_layer.w_par_deriv = tmp_w_par_deriv.reshape(before_layer.output_val.shape[1], curr_layer.neure_number, curr_layer.neure_number, order='F')
            curr_layer.b_par_deriv = curr_layer.error.T

    def jacobian(self):
        row = self.net_struct.x.shape[1]
        col = sum([layer.neure_number * (layer.neure_number + 1) for layer in self.net_struct.layers[1:]])
        j = np.zeros((row, col))
        index = 0
        for layer in self.net_struct.layers[1:]:
            w_col = layer.w_par_deriv.shape[0] * layer.w_par_deriv.shape[1]
            b_col = layer.b_par_deriv.shape[1]
            j[:, index:index + w_col] = layer.w_par_deriv.reshape(row, w_col)
            index += w_col
            j[:, index:index + b_col] = layer.b_par_deriv
            index += b_col
        return j

    def jjje(self):
        e = self.net_struct.y - self.net_struct.layers[-1].output_val
        e = e.T
        j = self.jacobian()
        jj = j.T.dot(j)
        je = -j.T.dot(e)
        return jj, je

    def lm(self):
        mu = self.mu
        beta = self.beta
        iteration = self.iteration
        tol = self.tol
        y = self.net_struct.y

        for i in range(iteration):
            self.forward()
            pred = self.net_struct.layers[-1].output_val
            pref = np.linalg.norm(y - pred) / len(y)

            if pref < tol:
                break

            self.backward()
            self.parDeriv()
            jj, je = self.jjje()

            while True:
                A = jj + mu * np.diag(np.ones(jj.shape[0]))
                delta_w_b = pinv(A).dot(je)
                old_net_struct = copy.deepcopy(self.net_struct)
                self.updataNetStruct(delta_w_b)

                self.forward()
                pred_new = self.net_struct.layers[-1].output_val
                pref_new = np.linalg.norm(y - pred_new) / len(y)

                if pref_new < pref:
                    mu /= beta
                    pref = pref_new
                    break
                else:
                    mu *= beta
                    self.net_struct = copy.deepcopy(old_net_struct)

            predictions = (pred_new > 0.5).astype(int)
            print("Iteration:", i, "Error:", pref, "Predictions:", predictions)

    def updataNetStruct(self, delta_w_b):
        index = 0
        for layer in self.net_struct.layers[1:]:
            w_num = layer.neure_number * layer.neure_number
            b_num = layer.neure_number
            w = delta_w_b[index:index + w_num].reshape(layer.neure_number, layer.neure_number, order='C')
            index += w_num
            b = delta_w_b[index:index + b_num]
            index += b_num
            layer.w += w
            layer.b += b