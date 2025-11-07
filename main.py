import numpy as np
import utils as u
import network_abstract as n

if __name__ == "__main__":

    '''
    pat = [[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19]]
    print(type(pat))
    w_lat = n.generate_w_exc(n_neurons=20, patterns=pat, p_connect = 0.5, mean_weight = 5, sd_weight = 2)
    print(w_lat)
    w_ie = n.generate_w_pv(n_neurons = 5, p_connect = 0.8, mean_weight = 1, sd_weight = 0.2)
    print(w_ie)
    '''

    random_pattern = u.generate_random_patterns(n_neurons = 40, pattern_size = 8, n_patterns = 5)
    sorted_pattern = [[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15],[16,17,18,19,20,21,22,23],[24,25,26,27,28,29,,30,31],[32,33,34,35,36,37,38,39]]
    # use np.where(n_neurons == 12) to find index of item 12 in np.array
    # -> for w_ee and w_ie the 1D has to be the same, meaning exc connect to inh of the same group
    # -> for w_ei the 1D cannot be the same, meaning inh connect to all exc that are of different group