import sys
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

paths = [os.path.dirname(os.path.abspath(__file__)),
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]
sys.path.extend(paths)

#from src.network_v1 import truncated_normal, truncated_lognormal
from utils import generate_random_patterns

def truncated_lognormal(mean, sigma, lower, upper, size):
    """Generate samples from a truncated lognormal distribution."""

    # Compute the corresponding normal mean μ
    mu = np.log(mean) - 0.5 * sigma ** 2

    # Compute truncation limits in normal space
    a, b = (np.log(lower) - mu) / sigma, (np.log(upper) - mu) / sigma

    # Generate samples from truncated normal, then exponentiate
    truncated_samples = stats.truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size)
    lognormal_samples = np.exp(truncated_samples)

    return lognormal_samples

def generate_w_exc(mat, n_neurons, patterns, p_connect, mean_weight, sd_weight):
    """
    Generates the EXC connectivity matrix.

    - Each neuron connects to 10% of all neurons.
    - 80% of connections are within the same cluster.
    - 20% of connections are to other clusters.
    - Weights follow a log-normal distribution.

    Parameters:
    - n (int): Total number of neurons.
    - patterns (list of lists): Each sublist contains indices of neurons in a cluster.
    - p_connect (float): Probability of connection (default 10% for EXC neurons).
    - intra_cluster_ratio (float): Fraction of connections within the same cluster (default 80%).
    - lognorm_mean (float): Mean for log-normal weight distribution.
    - lognorm_std (float): Standard deviation for log-normal weight distribution.

    Returns:
    - exc_matrix (np.ndarray): n × n connectivity matrix for excitatory neurons.
    """
    #w = np.zeros((n_neurons, n_neurons))  # Initialize connectivity matrix
    w = mat

    for pattern in patterns:

        for neuron in pattern:
            n_selected_intra = max(1, int(len(pattern) * p_connect))

            # Select intra-cluster connections (from the same cluster)
            selected_intra = np.random.choice(pattern, size=n_selected_intra, replace=False)

            # Assign log-normal weights
            weights_intra = truncated_lognormal(mean=mean_weight, sigma=sd_weight, size=n_selected_intra,
                                                          lower=0.1, upper=5.0)

            # Update the matrix
            w[neuron, selected_intra] = weights_intra

    np.fill_diagonal(w, 0)
    return w

def generate_w_ie(mat, n_exc_neurons, n_inh_neurons, neuron_range_exc, neuron_range_inh, pattern_exc_neurons, pattern_inh_neurons, p_connect, mean_weight, sd_weight):
    #weights = np.zeros((n_exc_neurons+n_inh_neurons, n_exc_neurons+n_inh_neurons))
    weights = mat
    random_values = np.random.rand(n_exc_neurons*n_inh_neurons)

    for e in np.arange(neuron_range_exc[0], neuron_range_exc[1]):
        for i in np.arange(neuron_range_inh[0], neuron_range_inh[1]):

            if np.where(pattern_inh_neurons == i)[0] == np.where(pattern_exc_neurons == e)[0]:

                w = (np.random.choice(random_values) < p_connect).astype(int)
                raw_weights = truncated_lognormal(mean=mean_weight, sigma=sd_weight, size=(n_exc_neurons+n_inh_neurons),
                                                lower=0.1, upper=10.0)

                weights[i, e] = w * raw_weights[e]
        
            print("we", weights[i,e], i, e)
            
            
    #np.fill_diagonal(w, 0)

    return weights

def generate_w_ei(mat, n_exc_neurons, n_inh_neurons, neuron_range_exc, neuron_range_inh, pattern_exc_neurons, pattern_inh_neurons, p_connect, mean_weight, sd_weight):
    #weights = np.zeros((n_exc_neurons+n_inh_neurons, n_exc_neurons+n_inh_neurons))
    weights = mat
    random_values = np.random.rand(n_exc_neurons*n_inh_neurons)

    for e in np.arange(neuron_range_exc[0], neuron_range_exc[1]):
        for i in np.arange(neuron_range_inh[0], neuron_range_inh[1]):

            if np.where(pattern_inh_neurons == i)[0] != np.where(pattern_exc_neurons == e)[0]:

                w = (np.random.choice(random_values) < p_connect).astype(int)
                raw_weights = truncated_lognormal(mean=mean_weight, sigma=sd_weight, size=(n_exc_neurons+n_inh_neurons),
                                                lower=0.1, upper=10.0)

                weights[e, i] = w * raw_weights[e] * (-1)
        
            # controls:
            # print("weights", weights[i,e], i, e)

    return weights


def connectivity_matrix(num_all_neurons, percentage_exc_neurons, num_patterns):

    # testing:
    # sorted_pattern_exc = [[0,1,2,3,4,5,6,7][8,9,10,11,12,13,14,15][16,17,18,19,20,21,22,23][24,25,26,27,28,29,30,31][32,33,34,35,36,37,38,39]]
    # sorted_pattern_inh = [[40,41],[42,43],[44,45],[46,47],[48,49]]

    exc_neurons = int(num_all_neurons*percentage_exc_neurons)
    inh_neurons = int(num_all_neurons-exc_neurons)

    assert exc_neurons % num_patterns == 0, f"Number of neurons is not compatible with number of patterns, exc_neurons: {exc_neurons}, inh_neurons: {inh_neurons}, num_patterns: {num_patterns}"
    assert inh_neurons % num_patterns == 0, f"Number of neurons is not compatible with number of patterns, exc_neurons: {exc_neurons}, inh_neurons: {inh_neurons}, num_patterns: {num_patterns}"

    whole_matrix = np.zeros((exc_neurons+inh_neurons, exc_neurons+inh_neurons))

    random_pattern_exc = generate_random_patterns(n_neurons = exc_neurons, neuron_range = (0, exc_neurons),
                                                  pattern_size = int(exc_neurons/num_patterns), n_patterns = num_patterns)
    
    random_pattern_inh = generate_random_patterns(n_neurons = inh_neurons, neuron_range = (exc_neurons, exc_neurons+inh_neurons),
                                                  pattern_size = int(inh_neurons/num_patterns), n_patterns = num_patterns)


    w_ee = generate_w_exc(mat=whole_matrix, n_neurons=exc_neurons, patterns=random_pattern_exc, p_connect = 1,
                          mean_weight = 1, sd_weight = 0.2)
    
    w_ie = generate_w_ie(mat=w_ee, n_exc_neurons=exc_neurons, n_inh_neurons=inh_neurons, neuron_range_exc = (0, exc_neurons),
                         neuron_range_inh = (exc_neurons, exc_neurons+inh_neurons),pattern_exc_neurons=random_pattern_exc,
                         pattern_inh_neurons=random_pattern_inh, p_connect=1, mean_weight=1, sd_weight=0.2)
    
    w_ei = generate_w_ei(mat=w_ie, n_exc_neurons=exc_neurons, n_inh_neurons=inh_neurons, neuron_range_exc = (0, exc_neurons),
                         neuron_range_inh = (exc_neurons, exc_neurons+inh_neurons),pattern_exc_neurons=random_pattern_exc,
                         pattern_inh_neurons=random_pattern_inh, p_connect=1, mean_weight=1, sd_weight=0.2)
    
    return w_ei, exc_neurons, inh_neurons


def plot_connectivity(w_ei, exc_neurons, inh_neurons):

    heatmap1 = sns.heatmap(data = w_ei, annot = False, fmt=".2f", linewidths=0, cmap = sns.color_palette("coolwarm", as_cmap=True))
    heatmap1.xaxis.tick_top()
    heatmap1.hlines(y = exc_neurons, xmin = 0, xmax = exc_neurons+inh_neurons, colors = "black")
    heatmap1.vlines(x = exc_neurons, ymin = 0, ymax = exc_neurons+inh_neurons, colors = "black")
    plt.show()



def generate_w_pv(n_neurons, p_connect, mean_weight, sd_weight):
    random_values = np.random.rand(n_neurons, n_neurons)
    w = (random_values < p_connect).astype(int)  # 6%

    raw_weights = truncated_lognormal(mean=mean_weight, sigma=sd_weight, size=(n_neurons, n_neurons),
                                      lower=0.1, upper=10.0)

    w = w * raw_weights
    #np.fill_diagonal(w, 0)

    return w * (-1)

def generate_w_som(n_neurons, patterns, p_connect,
                   mean_weight, sd_weight):

    w = np.zeros((n_neurons, n_neurons))  # Initialize connectivity matrix

    for pattern in patterns:
        other_neurons = list(set(range(n_neurons)) - set(pattern))  # Neurons outside the current pattern

        for neuron in pattern:
            # Select a subset of 8% neurons from other patterns
            n_selected = max(1, int(len(other_neurons) * p_connect))  # Ensure at least 1 connection
            selected_targets = np.random.choice(other_neurons, size=n_selected, replace=False)

            # Assign log-normal weights
            weights = truncated_lognormal(mean=mean_weight, sigma=sd_weight, size=n_selected,
                                          lower=0.1, upper=10.0)
            w[neuron, selected_targets] = weights  # Set weights in the matrix

    return w * (-1)

def build_connectivity(n_neurons=1000, n_patterns=10, pattern_size=100,
                       w_som_p=0.02, w_exc_p=0.015):
    patterns_fam = generate_random_patterns(n_neurons, pattern_size, n_patterns)

    w_pv_mean = 1
    w_pv_sd = 0.4
    w_pv_p = 0.06
    w_pv = generate_w_pv(n_neurons=n_neurons, p_connect=w_pv_p,
                         mean_weight=w_pv_mean, sd_weight=w_pv_sd
                         )
    #w_som_p = 0.02  # 0.11
    w_som_mean_weight = 1.0
    w_som_sd_weight = 0.5
    w_som = generate_w_som(n_neurons=n_neurons, patterns=patterns_fam, p_connect=w_som_p,
                           mean_weight=w_som_mean_weight, sd_weight=w_som_sd_weight
                           )
    #w_exc_p = 0.015  # 0.1
    w_exc_mean = 0.4
    w_exc_sd = 0.8
    w_exc = generate_w_exc(n_neurons=n_neurons, patterns=patterns_fam, p_connect=w_exc_p,
                           mean_weight=w_exc_mean, sd_weight=w_exc_sd)
    patterns_new = generate_random_patterns(n_neurons, pattern_size, n_patterns)

    return patterns_fam, patterns_new, w_exc, w_som, w_pv