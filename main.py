import numpy as np
import utils as u
import vectorized_network_abstract as n
import seaborn as sns
import matplotlib.pyplot as plt

# vectorize functions -> ask chat-gpt, numpy functions instead of loops

if __name__ == "__main__":

    w_ei, exc_neurons, inh_neurons = n.connectivity_matrix(num_all_neurons=100, percentage_exc_neurons=0.8, num_patterns=5)
    n.plot_connectivity(w_ei, exc_neurons, inh_neurons)
                

    


    # patterns must be converted to array
    # use np.where(n_neurons == 12) to find index of item 12 in np.array
    # -> for w_ee and w_ie the 1D has to be the same, meaning exc connect to inh of the same group
    # -> for w_ei the 1D cannot be the same, meaning inh connect to all exc that are of different group