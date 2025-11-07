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

    test = u.generate_random_patterns(n_neurons = 20, pattern_size = 5, n_patterns = 4)
    print(test)
    test_array = np.array(test)
    print(test_array)
    itemindex = np.where(test_array == 12)
    print(itemindex)
    