# Author (GitHub alias): coush001
"""utils module containing various support functions for the running of the main processes."""

# standard imports
import numpy as np
import random
import scipy

# pytorch imports
import torch
from torch.autograd import Variable
import torch.nn as nn

# segy load and save tool import
import segypy


def set_seed(seed):
    """
    Set random seed for all relevant randomisation tools.

    Parameters
    ----------
    seed : int
        Random seed value

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


def load_seismic(filename, inlines, xlines):
    """
    Load seismic amplitudes from .SEGY into numpy array.

    Parameters
    ----------
    filename : str
        File pathname for .SEGY file.
    inlines : list
        List in form [start, stop, step] for data inlines.
    xlines : list
        List in form [start, stop, step] for data inlines.

    Returns
    -------
    amplitude : array_like
        Array of seismic amplitudes.
    twt : array_like
        Array of twt range for data.
    """
    inl = np.arange(*inlines)
    crl = np.arange(*xlines)
    seis, header, trace_headers = segypy.readSegy(filename)
    amplitude = seis.reshape(header['ns'], inl.size, crl.size)
    lagtime = trace_headers['LagTimeA'][0] * -1
    twt = np.arange(lagtime, header['dt'] / 1e3 * header['ns'] + lagtime,
                    header['dt'] / 1e3)
    return amplitude, twt


def load_horizon(filename, inlines, xlines):
    """
    Load horizon from .txt into numpy array.

    Parameters
    ----------
    filename : str
        File pathname to .txt file
    inlines : list
        List in form [start, stop, step] for data inlines.
    xlines : list
        List in form [start, stop, step] for data inlines.

    Returns
    -------
    Numpy array of horizon depth.

    """
    inl = np.arange(*inlines)
    crl = np.arange(*xlines)
    hrz = np.recfromtxt(filename, names=['il', 'xl', 'z'])
    horizon = np.zeros((len(inl), len(crl)))
    for i, idx in enumerate(inl):
        for j, xdx in enumerate(crl):
            time = hrz['z'][np.where((hrz['il'] == idx) & (hrz['xl'] == xdx))]
            if len(time) == 1:
                horizon[i, j] = time

    return horizon


def interpolate_horizon(horizon):
    """
    Interpolates missing data in a horizon numpy array.

    Parameters
    ----------
    horizon : array_like
        horizon depth data.

    Returns
    -------
    Interpolated array.

    """

    points = []
    wanted = []
    for i in range(horizon.shape[0]):
        for j in range(horizon.shape[1]):
            if horizon[i, j] != 0.:
                points.append([i, j, horizon[i, j]])
            else:
                wanted.append([i, j])

    points = np.array(points)
    zs2 = scipy.interpolate.griddata(points[:, 0:2],
                                     points[:, 2],
                                     wanted,
                                     method="cubic")
    for p, val in zip(wanted, zs2):
        horizon[p[0], p[1]] = val

    return horizon


class VAE(nn.Module):
    """
    Pytorch implementation of vae.
    """
    def __init__(self, hidden_size, shape_in):
        """
        Define the architecture of VAE model.

        Parameters
        ----------
        hidden_size : int
            Size of the vae latent space.
        shape_in : array like
            Shape of the data input.
        """
        super(VAE, self).__init__()

        # Retrieve input dimension on each sample
        shape = shape_in[-1]

        assert shape % 4 == 0, 'input dimension for VAE must be factor of 4'

        # Specified reduction factor of each convolution, if layer number or stride is changed update this list!!
        reductions = [0.5, 0.5, 0.5]

        # number of channels after last convolution
        self.last_conv_channels = 34

        # find the resultant dimension post convolution layers
        post_conv = self.post_conv_dim(shape, reductions)

        # corresponding dimension for the input into linear fully connected layer
        self.linear_dimension = post_conv * self.last_conv_channels

        # Encoder
        self.conv1 = nn.Conv1d(2, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(32, self.last_conv_channels, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(self.linear_dimension, 128)

        # Latent space
        self.fc21 = nn.Linear(128, hidden_size)
        self.fc22 = nn.Linear(128, hidden_size)

        # Decoder
        self.fc3 = nn.Linear(hidden_size, 128)
        self.fc4 = nn.Linear(128, self.linear_dimension)
        self.deconv1 = nn.ConvTranspose1d(self.last_conv_channels, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose1d(32, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose1d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv1d(32, 2, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def post_conv_dim(self, in_shape, conv_reductions):
        """
        Calculates the dimension of the data at the end of convolutions.

        Parameters
        ----------
        in_shape : int
            Input dimension.
        conv_reductions : list
            List that specifies the reduction factor for each convolution, generally 1/stride of each layer.

        Returns
        -------
        int dimension post convolutions based on input dimension.
        """
        for i in conv_reductions:
            in_shape = int(np.ceil(
                in_shape * i))  #  calc the resultant size from each conv
        return in_shape

    def encode(self, x):
        """
        Encode the input into latent space variables.

        Parameters
        ----------
        x : array_like
            Input data array.

        Returns
        -------
            Mu and Logvar outputs
        """
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        h1 = self.relu(self.fc1(out))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        """
        Sample from the latent space variables

        Parameters
        ----------
        mu : multivariate mean
        logvar : log variance

        Returns
        -------
        Latent space sampled variables
        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        """
        Decode from latent space back into input dimension.

        Parameters
        ----------
        z : array_like
            Latent space representation.

        Returns
        -------
        Reconstructed data array.
        """
        h3 = self.relu(self.fc3(z))
        out = self.relu(self.fc4(h3))
        out = out.view(out.size(0), self.last_conv_channels,
                       int(self.linear_dimension / self.last_conv_channels))
        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        out = self.conv5(out)
        return out

    def forward(self, x):
        """
        __call__ function for the class.

        Parameters
        ----------
        x : array_like
            Model input data.

        Returns
        -------
        decode : array_like
            Reconstructed data in dimension of input.
        mu : array_like
            Latent space representation mean
        logvar : array_like
            Latent space representation variance
        z : array_like
            Latent space representation mean

        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


def loss_function(recon_x, x, mu, logvar, window_size, beta=1, recon_loss_method='mse'):
    """
    Loss function estimator for vae, referred to as the ELBO.

    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Parameters
    ----------
    recon_x : torch tensor
        reconstructed seismic traces
    x : torch tensor
        input seismic traces
    mu : torch tensor
        latent space mean
    logvar : torch tensor
        latent variance
    window_size : int
        input length of each trace
    beta : float
        beta value relevant for training a beta-vae
    recon_loss_method : str
        specifies the reconstruction loss technique to be used.

    Returns
    -------
    summed loss value for whole batch

    """
    # Mean squared error
    criterion_mse = nn.MSELoss(size_average=False)
    mse = criterion_mse(recon_x.view(-1, 2, window_size),
                        x.view(-1, 2, window_size))
    # p2 norm
    dist = torch.dist(recon_x.view(-1, 2, window_size),
                      x.view(-1, 2, window_size))

    # kl-divergance
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    if recon_loss_method == 'mse':
        recon_loss = mse

    elif recon_loss_method == 'dist':
        recon_loss = dist

    else:
        print('\nThe recon_loss_method chosen is not valid, "mse" will be used as default\n')
        recon_loss = mse

    return recon_loss + beta * KLD


def train(epoch, model, optimizer, train_loader, beta=1, recon_loss_method='mse'):
    """
    Trains a single epoch of the vae model.

    Parameters
    ----------
    epoch : int
        epoch number being trained
    model : torch.nn.module
        model being trained, here a vae
    optimizer : torch.optim
        optmizer used to train model
    train_loader : torch.utils.data.DataLoader
        data loader used for training
    beta : float
        beta parameter for the beta-vae
    recon_loss_method : str
        specifies the reconstruction loss technique

    Returns
    -------
    trains the model and returns training loss for the epoch

    """
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)

        optimizer.zero_grad()  # 1. zero grad after each batch
        recon_batch, mu, logvar, _ = model(data)  # 2. run training data through network
        loss = loss_function(recon_batch,  # 3. calculate the ELBO value on result
                             data,
                             mu,
                             logvar,
                             window_size=data.shape[-1],
                             beta=beta,
                             recon_loss_method=recon_loss_method)
        loss.backward()  # 4. calculated gradients

        # 'loss' is the SUM of all vector to vector losses in batch
        train_loss += loss.item()
        optimizer.step()  # Update parameters

    train_loss /= len(train_loader.dataset)
    return train_loss


def test(epoch, model, test_loader, beta=1, recon_loss_method='mse'):
    """
        Tests the vae model every epoch.

        Parameters
        ----------
        epoch : int
            epoch number being trained
        model : torch.nn.module
            model being trained, here a vae
        test_loader : torch.utils.data.DataLoader
            data loader used for training
        beta : float
            beta parameter for the beta-vae
        recon_loss_method : str
            specifies the reconstruction loss technique

        Returns
        -------
        test loss for the epoch

        """
    model.eval()
    test_loss = 0
    with torch.set_grad_enabled(False):
        for i, (data, _) in enumerate(test_loader):

            data = Variable(data)
            recon_batch, mu, logvar, _ = model(data)
            loss = loss_function(recon_batch,
                                 data,
                                 mu,
                                 logvar,
                                 data.shape[-1],
                                 beta=beta,
                                 recon_loss_method=recon_loss_method)
            test_loss += loss.item()

        test_loss /= len(test_loader.dataset)
    return test_loss


def forward_all(model, all_loader):
    """
    Run full training set through a trained model.

    Parameters
    ----------
    model : torch.nn.module
        model being run, here a vae.
    all_loader : torch.utils.data.DataLoader
        data loader used for running on model
    Returns
    -------
    torch tensors
        reconstructed data, latent dimension variables of data
    """
    model.eval()
    reconstructions, latents = [], []
    with torch.set_grad_enabled(False):
        for i, (data, _) in enumerate(all_loader):
            data = Variable(data)
            recon_batch, mu, logvar, z = model(data)
            reconstructions.append(recon_batch.cpu())
            latents.append(z.cpu())
    return torch.cat(reconstructions, 0), torch.cat(latents, 0)
