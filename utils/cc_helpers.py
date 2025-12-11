"""
A feature extractor class, an evaluation and a plotting function for channel charting.

@author: Sueda Taner
"""
#%%
import numpy as np
import torch

from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import trustworthiness

#%% Helper functions for feature extraction

def cellfree_multisc_bspace(H: torch.Tensor, A: int) -> torch.Tensor:
    """
    Transform an antenna domain channel matrix into beamspace domain.
    
    Parameters
    ----------
    H : torch.Tensor of size (U=num_users, B=num_BS_antennas, W=num_subcarriers)
        Channel matrices for each user, antenna and subcarrier.
    A : int
        Number of APs so that A * num_antenna_per_AP = B.

    Returns
    -------
    H : torch.Tensor of size (U=num_users, B=num_BS_antennas, W=num_subcarriers).
        Channel matrices for each user, antenna and subcarrier.    
    """
    temp = torch.swapaxes(H, -1, -2) # U W B
    temp2 = torch.reshape(temp, (H.shape[0], H.shape[-1], A, -1)) # (U, W, A, num_antenna_per_AP) 
    temp3 = torch.fft.fft(temp2, axis=-1, norm='ortho') # (U, W, A, num_antenna_per_AP) in beamspace
    temp4 = torch.reshape(temp3, (H.shape[0], H.shape[-1], -1)) # (U, W, B)
    Hhat = torch.swapaxes(temp4, -1, -2) # (U, B, W)
    return Hhat
    
def corr_in_dim3(H: torch.Tensor) -> torch.Tensor:
    """
    Compute the correlation in the third dimension of a 3D torch.Tensor.
    
    Parameters
    ----------
    H : torch.Tensor of size (U=num_users, B=num_BS_antennas, W=num_subcarriers)
        Channel matrices for each user, antenna and subcarrier.
        
    Returns
    -------
    H_cor : torch.Tensor of size (U=num_users, B=num_BS_antennas, W=num_subcarriers)
        Correlation of H in the third dimension.

    """
    H = H.numpy() # to use np's correlate function
    idx = H.shape[-1] -1
    H_cor = np.zeros(H.shape).astype(np.complex64)
    for r in range(H.shape[0]):
        for c in range(H.shape[1]):
            H_cor[r,c] = np.correlate(H[r,c],H[r,c],'full')[idx:]
    H_cor = torch.from_numpy(H_cor)
    return H_cor

#%%
class FeatureExtractor:
    """
    A class to represent feature extraction for channel charting.
    
    Attributes
    ----------
    feature_extraction_method : str
        Feature extraction method for channel charting.
    cc_method : str
        Dimensionality reduction method for channel charting.
        
    Methods
    -------
    feature_extract(H, A):
        Extracts features from the channel matrix H assuming A APs.

    """
    
    def __init__(self, feature_extraction_method='abs'):
        """
        Initialize FeatureExtractor object.
        
        Parameters
        ----------
        feature_extraction_method : str
            Feature extraction method for channel charting.
        """
        self.feature_extraction_method = feature_extraction_method

    def feature_extract(self, H: torch.Tensor, A=1) -> torch.Tensor:
        """
        Extract features for channel charting.

        Parameters
        ----------
        H : torch.Tensor
            Channel matrix.
        A : int
            Number of APs. The default is 1.

        Raises
        ------
        Exception
            if the object has an undefined feature extraction method.

        Returns
        -------
        X : torch.Tensor
            Features.

        """
        # Feature extraction methods:
        # 'abs', 'reim', 'beamspace abs', 'beamspace reim', 'cor abs', 'cor reim', 'beamspace + cor abs'
        
        if self.feature_extraction_method == 'abs':
            X = torch.reshape(H, (H.shape[0], -1))
            X = torch.abs(X) 
            
        elif self.feature_extraction_method == 'reim':
            X = torch.reshape(H, (H.shape[0], -1))
            X = torch.hstack((torch.real(X), torch.imag(X)))
            
        elif self.feature_extraction_method[:9] == 'beamspace':
            X = cellfree_multisc_bspace(H, A)
            X = torch.reshape(X, (X.shape[0], -1))
            
            if self.feature_extraction_method[-4:] == ' abs':
                X = torch.abs(X) 
            elif self.feature_extraction_method[-5:] == ' reim':
                X = torch.hstack((torch.real(X), torch.imag(X)))
            else:
                raise Exception('beamspace abs or real-imaginary?')
            
        elif self.feature_extraction_method[0:3] == 'cor':
            X = corr_in_dim3(H)
            X = torch.reshape(X, (X.shape[0], -1))
            if self.feature_extraction_method[-4:] == ' abs':
                X = torch.abs(X) 
            elif self.feature_extraction_method[-5:] == ' reim':
                X = torch.hstack((torch.real(X), torch.imag(X)))
            else:
                raise Exception('correlation abs or real-imaginary?')
            
        elif self.feature_extraction_method == 'beamspace + cor abs':
            X = cellfree_multisc_bspace(H, A)
            X = corr_in_dim3(X)
            X = torch.reshape(X, (X.shape[0], -1))
            X = torch.abs(X) 
            
        elif self.feature_extraction_method == 'none':
            X = torch.reshape(H, (H.shape[0], -1))

        else:
            raise Exception('Undefined feature extraction method')
        
        # Normalization applies to all feature extraction methods
        X = X / torch.linalg.norm(X, ord=2, dim=-1, keepdim=True) # normalize    
        return X

#%% Evaluate a channel chart
    
def compute_KS_from_D(d: np.array, d_tilde: np.array):
    # Kruskal stress
    beta = np.sum(d * d_tilde) / np.linalg.norm(d_tilde, 'fro') ** 2
    ks = np.linalg.norm(d - beta * d_tilde, 'fro') \
            / np.linalg.norm(d, 'fro')
    return ks

def compute_RD_from_D(d: np.array, d_tilde: np.array, metric_param=-1):
    # Rajski distance
    if metric_param == -1:
        num_bins = 20
    else:
        num_bins = metric_param
    x = d.flatten()
    y = d_tilde.flatten()
    Px, bin_edges_x = np.histogram(x, bins=num_bins)
    Px = Px / len(x)

    Py, bin_edges_y = np.histogram(y, bins=num_bins)
    Py = Py / len(y)

    Pxy, xedges, yedges = np.histogram2d(x, y, bins=num_bins)
    Pxy = Pxy / len(x)

    idcs = np.nonzero(Pxy)
    H = - np.sum(Pxy[idcs] * np.log2(Pxy[idcs]))
    I = np.sum(Pxy[idcs] * np.log2(Pxy[idcs] / (Px[idcs[0]] * Py[idcs[1]])))

    assert H != 0
    rd = 1 - I / H
    return rd    


def evaluate_cc(X: np.array, Y: np.array, metric='TW-CT', plot_eval=False, metric_param=-1):
    """
    Evaluate the quality of a latent space (specifically, the channel chart).
    
    Parameters
    ----------
    X : np.array of size (U, dimension=2 or 3)
        True positions of the UEs in the original space.
    Y : np.array of size (U, dimension=2 or 3)
        Positions of the UEs in the channel chart.
    metric : str, optional
        The evaluation metric. The default is 'TW-CT'. Can be 'KS' or 'RD' too.
    plot_eval : bool, optional
        True to plot TW-CT values. The default is False.
    metric_param : TYPE, optional
        DESCRIPTION. The default is -1.

    Raises
    ------
    Exception
        for undefined evaluation metrics.

    Returns
    -------
        An np.array if the metric is 'TW-CT' and a number if the metric is 'KS' or 'RD'
    """
    
    if metric == 'TW-CT':  # trustworthiness - continuity
        # t0 = time.time()
        tw = trustworthiness(X, Y, n_neighbors = int(0.05 * Y.shape[0]))    
        # t1 = time.time()
        # print(f'TW took {t1-t0:.3f}')
        # Continuity is trustworthiness with swapped inputs
        # t0 = time.time()
        ct = trustworthiness(Y, X, n_neighbors = int(0.05 * Y.shape[0]))
        # t1 = time.time()
        # print(f'CT took {t1-t0:.3f}')
        return tw, ct
    
    else:
        # Pair up calculating KS and RD to potentially avoid calculating distance matrices twice 
        # Calculate pairwise distances
        d = euclidean_distances(X, X)  # distance between the true positions
        d_tilde = euclidean_distances(Y, Y)  # distance between the points in the latent space

        if metric == 'KS': 
            # Kruskal stress
            return compute_KS_from_D(d, d_tilde)
        
        elif metric == 'RD':  
            # Rajski distance
            return compute_RD_from_D(d, d_tilde)
        
        elif metric == 'KS-RD': 
            ks = compute_KS_from_D(d, d_tilde)
            rd = compute_RD_from_D(d, d_tilde)
            return ks, rd
        else:
            raise Exception('Undefined evaluation metric')


#%% Plot a channel chart

def plot_chart(X: np.array, colors: np.array, normalize: bool, anchors=None, is_meters=True, lims=None, title=None, X_ = None):
    """
    Plot the channel chart with given colors.

    Parameters
    ----------
    X : np.array
        Positions of the UEs in the channel chart.
    colors : np.array
        RGB colors of the UEs.
    normalize : bool
        Whether the channel chart should be normalized to have zero mean and unit variance. 
    anchors : np.array
        Anchor points. The default is None.
    is_meters : bool, optional
        Whether the x- and y-labels should denote "(m)". The default is True.
    lims : np.array
        x- and y-axis limits. The default is None.
    title : str
        Title of the plot. The default is None.

    Returns
    -------
    fig : TYPE
        The figure.

    """
    fig = plt.figure()
    
    if normalize:
        X = (X - np.mean(X,0,keepdims=True)) / np.std(X,0,keepdims=True)
    
    if X.shape[1] == 1:
        ax = fig.add_subplot()
        ax.scatter(range(X.shape[0]), X, c=colors)
        ax.set_xlabel('index'), ax.set_ylabel('x')
        if is_meters: ax.set_ylabel('x [m]')
        if anchors is not None: ax.scatter(anchors[:,0], marker='^')
    elif X.shape[1] == 2:
        ax = fig.add_subplot()
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=5)
        if X_ is not None:
            ax.scatter(X_[:, 0], X_[:, 1], c="b", s=5)
        ax.set_xlabel('x'), ax.set_ylabel('y')
        if is_meters: ax.set_xlabel('x [m]'), ax.set_ylabel('y [m]')
        ax.axis('equal')
        if lims is not None:
            ax.set_xbound(lims[0,0],lims[0,1])
            ax.set_ybound(lims[1,0],lims[1,1])
            # ax.set(xlim=(lims[0,0], lims[0,1]), ylim=(lims[1,0], lims[1,1]))
        ax.axis('equal')
        # ax.set_aspect('equal')
        if anchors is not None: ax.scatter(anchors[:,0], anchors[:,1], marker='^')
    else:  # X.shape[1] == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors)
        ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
        if is_meters: ax.set_xlabel('x [m]'), ax.set_ylabel('y [m]'), ax.set_zlabel('z')
        if anchors is not None: ax.scatter(anchors[:,0], anchors[:,1], anchors[:,2], marker='^')
    
    ax.grid(True)
    if title is not None:
        plt.title(title)
    return fig

