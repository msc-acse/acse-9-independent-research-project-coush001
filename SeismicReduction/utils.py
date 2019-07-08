"""
utils module containing various support functions for the running of the main processes

"""


# standard imports
import numpy as np
import random
import scipy

# pytorch
import torch
from torch.autograd import Variable
import torch.nn as nn

# segy load and save tool
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

    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled = False

    return True


def load_seismic(filename, inlines=[1300, 1502, 2], xlines=[1500, 2002, 2]):
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


def load_horizon(filename, inlines=[1300, 1502, 2], xlines=[1500, 2002, 2]):
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
    Interpolated array for full field coverage.

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
    Pytorch compatible vae model implementation.
    """
    def __init__(self, hidden_size, shape_in):
        """
        Define the architecture of VAE model.

        Parameters
        ----------
        hidden_size : int
            Size of the vae latent space.
        shape_in : int
            Size of the input dimension.
        """
        super(VAE, self).__init__()

        #  Architecture paramaters
        shape = shape_in[-1]

        assert shape % 4 == 0, 'input dimension for VAE must be factor of 4'
        reductions = [0.5, 0.5,
                      0.5]  # specified reduction factor of each convolution

        self.last_conv_channels = 34  # number of channels after last convolution

        # find the resultant dimension post convolution layer processing
        post_conv = self.post_conv_dim(shape, reductions,
                                       self.last_conv_channels)

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
        self.deconv1 = nn.ConvTranspose1d(self.last_conv_channels,
                                          32,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1)
        self.deconv2 = nn.ConvTranspose1d(32,
                                          32,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1)
        self.deconv3 = nn.ConvTranspose1d(32,
                                          32,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1)
        self.conv5 = nn.Conv1d(32, 2, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def post_conv_dim(self, in_shape, conv_reductions, last_conv_channels):
        """
        Calculates the dimension of the data at the end of convolutions.

        Parameters
        ----------
        in_shape : int
            Input dimension.
        conv_reductions : list
            List that specifies the reduction factor for each convolution, generally 1/stride of each layer.
        last_conv_channels : int
            Value of number of output channels of last convolution.

        Returns
        -------
        int of dimension post convolutions.
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
        Latent space representation.
        """
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        h1 = self.relu(self.fc1(out))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
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


def loss_function(recon_x, x, mu, logvar, window_size):
    criterion_mse = nn.MSELoss(size_average=False)
    #     print('in loss func, window_size:', window_size)
    #     print('in loss func, x shape:', x.shape, recon_x.shape)
    MSE = criterion_mse(recon_x.view(-1, 2, window_size),
                        x.view(-1, 2, window_size))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD


# Function to perform one epoch of training
def train(epoch, model, optimizer, train_loader, cuda=False, log_interval=10):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        #         print(data.shape)

        if cuda:
            data = data.cuda()

        optimizer.zero_grad()
        recon_batch, mu, logvar, _ = model(data)
        #         print('In train, data shape:', print(data.shape))
        loss = loss_function(recon_batch,
                             data,
                             mu,
                             logvar,
                             window_size=data.shape[-1])
        loss.backward()
        train_loss += loss.item() * data.size(0)
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
    return train_loss


# Function to perform evaluation of data on the model, used for testing
def test(epoch, model, test_loader, cuda=False, log_interval=10):
    model.eval()
    test_loss = 0
    with torch.set_grad_enabled(False):
        for i, (data, _) in enumerate(test_loader):
            if cuda:
                data = data.cuda()
            data = Variable(data)
            recon_batch, mu, logvar, _ = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar,
                                       data.shape[-1]).item() * data.size(0)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


# Function to forward_propagate a set of tensors and receive back latent variables and reconstructions
def forward_all(model, all_loader, cuda=False):
    model.eval()
    reconstructions, latents = [], []
    with torch.set_grad_enabled(False):
        for i, (data, _) in enumerate(all_loader):
            if cuda:
                data = data.cuda()
            data = Variable(data)
            recon_batch, mu, logvar, z = model(data)
            reconstructions.append(recon_batch.cpu())
            latents.append(z.cpu())
    return torch.cat(reconstructions, 0), torch.cat(latents, 0)
