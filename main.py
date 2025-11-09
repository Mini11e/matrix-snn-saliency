import numpy as np
import utils as u
import network_abstract as n

if __name__ == "__main__":

    exc_neurons = 40
    inh_neurons = 10

    whole_matrix = np.zeros((exc_neurons+inh_neurons, exc_neurons+inh_neurons))

    random_pattern_exc = u.generate_random_patterns(n_neurons = exc_neurons, neuron_range = (0, exc_neurons), pattern_size = 8, n_patterns = 5)
    sorted_pattern_exc = [[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15],[16,17,18,19,20,21,22,23],[24,25,26,27,28,29,30,31],[32,33,34,35,36,37,38,39]]
    random_pattern_inh = u.generate_random_patterns(n_neurons = inh_neurons, neuron_range = (exc_neurons, exc_neurons+inh_neurons), pattern_size = 2, n_patterns = 5)
    sorted_pattern_inh = [[40,41],[42,43],[44,45],[46,47],[48,49]]

    w_ee = n.generate_w_exc(mat=whole_matrix, n_neurons=exc_neurons, patterns=sorted_pattern_exc, p_connect = 1, mean_weight = 1, sd_weight = 0.2)
    w_ie = n.generate_w_ie(mat=w_ee, n_exc_neurons=exc_neurons, n_inh_neurons=inh_neurons, neuron_range_exc = (0, exc_neurons), neuron_range_inh = (exc_neurons, exc_neurons+inh_neurons),pattern_exc_neurons=sorted_pattern_exc, pattern_inh_neurons=sorted_pattern_inh, p_connect=1, mean_weight=1, sd_weight=0.2)
    w_ei = n.generate_w_ei(mat=w_ie, n_exc_neurons=exc_neurons, n_inh_neurons=inh_neurons, neuron_range_exc = (0, exc_neurons), neuron_range_inh = (exc_neurons, exc_neurons+inh_neurons),pattern_exc_neurons=sorted_pattern_exc, pattern_inh_neurons=sorted_pattern_inh, p_connect=1, mean_weight=1, sd_weight=0.2)
    print(w_ei)


    # use np.where(n_neurons == 12) to find index of item 12 in np.array
    # -> for w_ee and w_ie the 1D has to be the same, meaning exc connect to inh of the same group
    # -> for w_ei the 1D cannot be the same, meaning inh connect to all exc that are of different group