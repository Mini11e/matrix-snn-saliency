import numpy as np
import utils as u
import network_abstract as n
import seaborn as sns
import matplotlib.pyplot as plt

# make alternative functions: neurons can be in multiple patterns, inh neurons can connect back to neuron
# of their group if it is also in another group

# vectorize functions -> ask chat-gpt

if __name__ == "__main__":


    #w_ei, exc_neurons, inh_neurons = n.connectivity_matrix(num_all_neurons=100, percentage_exc_neurons=0.8, num_patterns=5)
    #n.plot_connectivity(w_ei, exc_neurons, inh_neurons)

    w_ie = np.zeros((10,10))
    
    w_ei = n.generate_w_ei(mat=w_ie, n_exc_neurons=6, n_inh_neurons=4, neuron_range_exc = (0, 6),
                         neuron_range_inh = (6, 10),pattern_exc_neurons=[[0,1,2],[2,4,5]],
                         pattern_inh_neurons=[[6,7],[8,9]], p_connect=1, mean_weight=1, sd_weight=0.2)
    


    # use np.where(n_neurons == 12) to find index of item 12 in np.array
    # -> for w_ee and w_ie the 1D has to be the same, meaning exc connect to inh of the same group
    # -> for w_ei the 1D cannot be the same, meaning inh connect to all exc that are of different group