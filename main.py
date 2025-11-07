import numpy as np
import utils as u
import network_abstract as n

if __name__ == "__main__":

    
    random_pattern_exc = u.generate_random_patterns(n_neurons = 40, neuron_range = (0, 40), pattern_size = 8, n_patterns = 5)
    sorted_pattern_exc = [[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15],[16,17,18,19,20,21,22,23],[24,25,26,27,28,29,30,31],[32,33,34,35,36,37,38,39]]
    random_pattern_inh = u.generate_random_patterns(n_neurons = 10, neuron_range = (40, 50), pattern_size = 2, n_patterns = 5)
    sorted_pattern_inh = [[40,41],[42,43],[44,45],[46,47],[48,49]]

    w_ee = n.generate_w_exc(n_neurons=40, patterns=sorted_pattern_exc, p_connect = 1, mean_weight = 1, sd_weight = 0.2)
    w_ie = n.generate_w_ie(n_exc_neurons=40, n_inh_neurons=10, pattern_exc_neurons=sorted_pattern_exc, pattern_inh_neurons=sorted_pattern_inh, p_connect=1, mean_weight=1, sd_weight=0.2)
    
    #w_ei =


    # use np.where(n_neurons == 12) to find index of item 12 in np.array
    # -> for w_ee and w_ie the 1D has to be the same, meaning exc connect to inh of the same group
    # -> for w_ei the 1D cannot be the same, meaning inh connect to all exc that are of different group