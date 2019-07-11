""" core module contains the main classes to run unsupervised machine learning on seismic data """

# standard imports
import numpy as np
import matplotlib.pyplot as plt
import scipy
import copy
import random
import os
import datetime

# Machine learning tools
import umap
from sklearn.linear_model import LinearRegression
import torch
import torch.utils.data
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import TensorDataset
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA

# File load and save imports
from .utils import *

# live loss plots
from livelossplot import PlotLosses


class DataHolder:
    """
    Class to load and store the the suite of data for a single 'field'.

    Attributes
    ----------
    near : array_like
        Three dimensional numpy array of the near-offset amplitudes.

    far : array_like
        Three dimensional numpy array of the far-offset amplitudes.

    twt : array_like
        One dimensional array describing the two way times relating to the amplitude data.

    horizon : array_like
        Two dimensional array containing interpolated horizon depths.

    """
    def __init__(self, field_name, inlines, xlines):
        """
        Initialisation function to set the name and specify data ranges.


        Parameters
        ----------
        field_name : string
            The name of the dataset, for example the field name "Glitne".
        inlines : array_like
            The range and step of the inlines of the seismic data,
             formatted as [start, stop, step].
        xlines : array_like
            The range and step of the cross lines of the seismic data,
             formatted as [start, stop, step].
        """
        # User input attributes
        self.field_name = field_name
        self.inlines = inlines
        self.xlines = xlines

        # KEY data for processing
        self.near = None
        self.far = None
        self.twt = None
        self.horizon = None

        self.wells = {}

    def add_near(self, fname):
        """
        Load the near offset amplitudes from .SEGY file.

        Parameters
        ----------
        fname : string
            Pathname to the .SEGY near offset file.

        Returns
        -------
        None

        """
        self.near, twt = load_seismic(fname,
                                      inlines=self.inlines,
                                      xlines=self.xlines)
        self.twt = twt
        return

    def add_far(self, fname):
        """
        Load the far offset amplitudes from .SEGY file.

        Parameters
        ----------
        fname : string
            Pathname to the .SEGY far offset file.

        Returns
        -------
        None

        """
        self.far, twt = load_seismic(fname,
                                     inlines=self.inlines,
                                     xlines=self.xlines)
        assert (self.twt == twt
                ).all, "This twt does not match the twt from the previous segy"
        return

    def add_horizon(self, fname):
        """
        Load and interpolate horizon depths from .txt file.

        Parameters
        ----------
        fname : string
            Pathname to the .txt horizon file.

        Returns
        -------
        None

        """
        self.horizon = interpolate_horizon(
            load_horizon(fname, inlines=self.inlines, xlines=self.xlines))
        return

    def add_well(self, well_id, well_i, well_x):
        """
        Method to add well to a dictionary.

        Parameters
        ----------
        well_id : str
            Identification for the well.
        well_i : int
            Position of well in inlines.
        well_x : int
            Position of well in the cross-lines.

        Returns
        -------
        None

        """
        self.wells[well_id] = [well_i, well_x]
        return


class Processor:
    """
    Takes a dataset and performs various processing routines to produce output for analysis.

    Attributes
    ----------
    raw : array_like
        Contains the raw seismic amplitudes derived from a DataHolder object.
    twt : array_like
        Two way time, derived from DataHolder.
    out : array_like
        The output attribute used for each processing routine.
        Stored in list form: [near, far].
    attributes : dict
        A dictionary to contain the array of attributes of the seismic data.
        Examples are horizon depth and AVO derived fluid factor
    """
    def __init__(self, dataholder):
        """
        Initialise the processor with data from a DataHolder object.

        Parameters
        ----------
        dataholder : object
            A DataHolder object which the processor uses as source for processing.
        """
        self.raw = [dataholder.far, dataholder.near]
        self.twt = dataholder.twt
        self.out = None

        # attributes
        self.attributes = {'horizon_raw': dataholder.horizon}

    def flatten(self, data, top_add=12, below_add=52):
        """
        Flatten the seismic data along the horizon.

        Parameters
        ----------
        data : array_like
            List amplitudes in the form [far, near]
            Each set expected to be of shape (twt,x1,x2)
        top_add : int
            Defines the number of samples to maintain above the horizon.
        below_add : int
            Defines the number of samples to maintain below the horizon.

        Returns
        -------
        Flattened near and far amplitudes in list form: [near, far].
        Shape (twt, x1,x2).

        """
        out = []
        horizon = self.attributes['horizon_raw']

        # input data = [near(twt, x1, x2),far(twt, x1, x2)]
        for offset in data:
            # create output trace shape for each set in shape: (twt, x1, x2)
            traces = np.zeros(
                (top_add + below_add, horizon.shape[0], horizon.shape[1]))
            for i in range(horizon.shape[0]):
                #  find the corresponding nearest index of the horizon from nearest twt value
                hrz_idx = [
                    np.abs(self.twt - val).argmin() for val in horizon[i, :]
                ]
                for j in range(horizon.shape[1]):
                    # place the amplitudes from above:below horizon into 1st index
                    traces[:, i, j] = offset[hrz_idx[j] - top_add:hrz_idx[j] + below_add, i, j]
            out.append(traces)

        return out  # list of far and near, flattened amplitudes shape (twt, x1,x2)

    def trim(self, data, top_index=0, bottom_index=232):
        """
        Without flattening retrieve a vertically cropped segment of the seismic block.

        Parameters
        ----------
        data : array_like
            List amplitudes in the form [far, near]
            Each set expected to be of shape (twt,x1,x2)
        top_index : int
            Index for top of data.
        bottom_index : int
            Index for the bottom of the data

        Returns
        -------
        Seismic amplitudes from top_index to bottom_index
        """
        out = []

        # input data = [near(twt, x1, x2),far(twt, x1, x2)]
        for offset in data:
            traces = offset[top_index:bottom_index, :, :]
            out.append(traces)

        return out  # list of far and near, cropped amplitudes shape (twt, x1,x2)

    def roll_axis(self, data):
        """
        Transform the data axis to place two way times along the third axis.

        Parameters
        ----------
        data : array_like
            List amplitudes in the form [far, near]
            Each set expected to be of shape (twt,x1,x2)

        Returns
        -------
        data : array_like
            List amplitudes in the form [far, near]
            Each set expected to be of shape (x1,x2,twt)

        """
        # input should be (twt,x1,x2)
        for i in range(len(data)):
            data[i] = np.transpose(data[i], (1, 2, 0))
        # output (x1,x2,twt)
        return data

    def normalise(self, data):
        """
        Normalise data around a given 'well' sample area.

        Parameters
        ----------
        data : array_like
            List amplitudes in the form [far, near]
            Each set expected to be of shape (x1,x1,twt)

        Returns
        -------
        Normalised data arrays

        """
        well_i = 38
        well_x = 138
        out = []
        for i in data:
            # well_variance = np.mean(
            #     np.std(i[well_i - 2:well_i + 1, well_x - 2:well_x + 1], 2))
            # i /= well_variance
            variance = np.mean(
                np.std(i, 2))
            i /= variance
            out.append(i)

        return out

    def to_2d(self, data):
        """
        Flatten data from two spatial dimensions to one.

        Parameters
        ----------
        data : array_like
            List amplitudes in the form [far, near]
            Each set expected to be of shape (x1,x1,twt)

        Returns
        -------
        list form [far, near]
        shape of (number_traces, twt)


        """
        return [i.reshape(-1, data[0].shape[-1]) for i in data]

    def stack_traces(self, data):
        """
        Stack near and far traces along the second axis of a new array.

        Parameters
        ----------
        data : array_like
            List amplitudes in the form [far, near]
            Each set expected to be of shape (number_traces,twt)

        Returns
        -------
        Single three dimensional array, shape (number_traces, 2, twt).
        Second dimension references the far or near offset.

        """
        return np.stack([data[0], data[1]], axis=1)

    def run_AVO(self):
        """
        Run the avo analysis on the processed data.

        Will add fluid factor to the attributes dictionary.

        """
        x_avo = self.out[1]
        y_avo = self.out[0] - self.out[1]

        lin_reg = LinearRegression(fit_intercept=False,
                                   normalize=False,
                                   copy_X=True,
                                   n_jobs=1)
        lin_reg.fit(x_avo.reshape(-1, 1), y_avo.reshape(-1, 1))

        self.attributes['FF'] = y_avo - lin_reg.coef_ * x_avo

    def condition_attributes(self):
        """
        Works horizon attributes into the shape: (number_traces)
        """
        #  flatten horizon array
        horizon = self.attributes['horizon_raw']
        self.attributes['horizon'] = horizon.reshape(horizon.shape[0] *
                                                     horizon.shape[1])

        #  condense fluid factor to min of each trace
        self.attributes['FF'] = np.min(self.attributes['FF'], 1)

    def __call__(self, flatten=[False, 0, 0], crop=[False, 0, 0], normalise=False):
        """
        Call function controls the full processing routine.

        A new call can be made repeatedly on an existing object as many times as needed.
        The routine will be run from the raw data each time and output a processed array and
        attributes. Only flatten OR crop OR neither is accepted NOT both.

        Parameters
        ----------
        flatten : list
            specifying flattening parameters:
                flatten[0] : Bool
                flatten[1] : int specifying top add
                flatten[2] : int specifying below add
        crop : list
            specifying crop parameters:
                crop[0] : Bool
                crop[1] : int specifying top add
                crop[2] : int specifying below add


        normalise : list
                List specifying normalisation parameters:
                flatten[0] : Bool specifying whether to run flattening method
                flatten[1] : int specifying top add
                flatten[2] : int specifying below add


        Returns
        -------
        List of output array and attributes
        Output array is three dimensional numpy array, of shape (number_samples, 2, twt)
        Values of attribute dict in shape (number_samples)

        """
        self.out = copy.copy(self.raw)  # Set out attribute to raw data

        if flatten[0]:  # Flatten samples with top add, and bellow add
            self.out = self.flatten(self.out, flatten[1], flatten[2])

        elif crop[0]:  # trim seismic from flatten[1] to flatten[2]
            self.out = self.trim(self.out, crop[1], crop[2])

        self.out = self.roll_axis(self.out)  # Reorder axis of data to [x1,x2,twt]

        if normalise:  # Normalise samples
            self.out = self.normalise(self.out)

        # flatten to 2d (traces, twt)
        self.out = self.to_2d(self.out)

        # Find fluid factor, add to attributes
        self.run_AVO()

        # condition attributes to 1d arrays
        self.condition_attributes()

        #  Stack the traces for output
        self.out = self.stack_traces(self.out)
        print('Processor has created an output with shape: ', self.out.shape)

        return [self.out, self.attributes]


class ModelAgent:
    """
    Parent class to handle common functionality of the different unsupervised modelling processes.

    Attributes
    ----------
    input : array_like
        Three dimensional processed input array of seismic data to be analysed.
    attributes : dict
        Seismic trace attributes used for plotting. Example : fluid factor.
    input_dimension : int
        The dimension of the each input trace, used for VAE model initialisation.
    embedding : array_like
        Unsupervised low-dimension representation of the input.
    two_dimensions : array_like
        The two dimensional representation used for visualisation, either direct from model or
        represented via umap algorithm
    loaded_model : bool
        Flags whether this object is just used for loading and running a model
        If true, no training routine will be run.

    """
    def __init__(self, data):
        """
        Initialisation of the parent class, accessed via a specific model initialisation.

        Parameters
        ----------
        data : list
            An output from the processor. Contains the seismic data for analysis and corresponding attributes.
        """
        self.input = data[0]
        self.attributes = data[1]
        self.input_dimension = self.input.shape[-1]
        self.embedding = None  # intermediate embedding from model
        self.two_dimensions = None  # final output for visualisation
        self.loaded_model = False  # define whether this object is for loading or training a model
        print("ModelAgent initialised")

    def concat(self):
        """
        Reshapes input from three to two dimensions, collapsing far and near data into one dimension.

        Returns
        -------
        Modifies input from three to two dimensional array.
        """
        return self.input.reshape(-1, 2 * self.input_dimension)

    def to_2d(self, umap_neighbours=50, umap_dist=0.001, verbose=False):
        """
        Takes abritrary dimension of embedding and converts to two dimensions via umap.

        Parameters
        ----------
        umap_neighbours : int
            Control over local vs global structure representation. see UMAP class for more detailed description.
        umap_dist : float
            Control on minimum distance of output representations, see again UMAP class for more detailed description.
        verbose : bool
            Control on whether to print UMAP verbose output.

        Returns
        -------
        embedding : array_like
            Two dimensional representation of the model embedding.
        """

        if self.embedding.shape[1] == 2:
            print('NOTE: embedding already reduced to 2D latent space, UMAP will not be run')
            self.two_dimensions = self.embedding

        else:  # run umap
            print('\n2D UMAP representation of {} embedding initialised:'.format(self.name))
            print('\tInput dimension:', self.embedding.shape)
            transformer = umap.UMAP(n_neighbors=umap_neighbours,
                                    min_dist=umap_dist,
                                    metric='correlation',
                                    verbose=verbose).fit(self.embedding)
            self.two_dimensions = transformer.transform(self.embedding)
            print("\t2-D UMAP representation complete\n")

        return

    def save_nn(self, path):
        """
        Save a trained neural network to file.

        Parameters
        ----------
        path : str
            File path to save model

        Returns
        -------
        None
        """
        torch.save(self.model, path)

    def load_nn(self, path):
        """
        Load a saved neural network from file.

        Parameters
        ----------
        path : str
            File path to load model from

        Returns
        -------

        """
        self.loaded_model = True
        self.model = torch.load(path)


class PcaModel(ModelAgent):
    """
    Runs the PCA algorithm to reduce dimensionality of input to chosen dimension.
    """
    def __init__(self, data):
        super().__init__(data)
        self.name = 'PCA'

    def reduce(self, n_components=2):
        """
        Controller method for the dimensionality reduction routine.

        Parameters
        ----------
        n_components : int
            Number of dimensions to reduce to.

        Returns
        -------
        Modifies embedding attribute via generation of the low dimensional representation.

        """
        concat_near_far = self.concat()  # collapse the near far data into 1 dimension

        pca_model = PCA(n_components=n_components)
        p_components = pca_model.fit_transform(concat_near_far)

        self.embedding = p_components

    def save_nn(self, name):
        raise Exception('Method is not appropriate for this type of model - No Neural Net!')

    def load_nn(self, name):
        raise Exception('Method is not appropriate for this type of model - No Neural Net!')


class UmapModel(ModelAgent):
    """
    Runs the UMAP algorithm to reduce dimensionality of input to two dimensions.
    """
    def __init__(self, data):
        super().__init__(data)
        self.name = 'UMAP'

    def reduce(self):
        """
        Prepare self.embedding for umap_2d method.

        Concatenate the near/far offset amplitudes to generate 2d array for the umap model.

        """
        self.embedding = self.concat()  # collapse the near far data into 1 dimension

    def save_nn(self, name):
        raise Exception('Method is not appropriate for this type of model - No Neural Net!')

    def load_nn(self, name):
        raise Exception('Method is not appropriate for this type of model - No Neural Net!')


class VaeModel(ModelAgent):
    """
    Runs the VAE model to reduce the seismic data to an arbitrary sized dimension, visualised in 2 via UMAP.
    """
    def __init__(self, data):
        super().__init__(data)
        self.name = 'VAE'

    def create_dataloader(self, batch_size=32):
        """
        Create pytorch data loaders for use in vae training, testing and running.

        Parameters
        ----------
        batch_size : int
            Size of data loader batches.

        Returns
        -------
        Modifies object data loader attributes.

        """
        # create torch tensor
        assert self.input.shape[1] == 2, 'Expected a three dimensional input with 2 channels'
        X = torch.from_numpy(self.input).float()

        # Create a stacked representation and a zero tensor so we can use the standard Pytorch TensorDataset
        y = torch.from_numpy(np.zeros((X.shape[0], 1))).float()

        split = ShuffleSplit(n_splits=1, test_size=0.5)
        for train_index, test_index in split.split(X):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

        train_dset = TensorDataset(X_train, y_train)
        test_dset = TensorDataset(X_test, y_test)
        all_dset = TensorDataset(X, y)

        kwargs = {'num_workers': 1, 'pin_memory': True}
        self.train_loader = torch.utils.data.DataLoader(train_dset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        **kwargs)
        self.test_loader = torch.utils.data.DataLoader(test_dset,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       **kwargs)
        self.all_loader = torch.utils.data.DataLoader(all_dset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      **kwargs)

    def train_vae(self, epochs=5, hidden_size=8, lr=1e-2, recon_loss_method='mse'):
        """
        Handles the training of the vae model.

        Parameters
        ----------
        epochs : int
            Number of complete passes over the whole training set.
        hidden_size : int
            Size of the latent space of the vae.
        lr : float.
            Learning rate for the vae model training.
        recon_loss_method : str
            Method for reconstruction loss calculation

        Returns
        -------
        None

        """
        set_seed(42)  # Set the random seed
        self.model = VAE(hidden_size,
                         self.input.shape)  # Inititalize the model

        # Create a gradient descent optimizer
        optimizer = optim.Adam(self.model.parameters(),
                               lr=lr,
                               betas=(0.9, 0.999))

        if self.plot_loss:
            liveloss = PlotLosses()
            liveloss.skip_first = 0
            liveloss.figsize = (16, 10)

        # Start training loop
        for epoch in range(1, epochs + 1):
            tl = train(epoch,
                       self.model,
                       optimizer,
                       self.train_loader,
                       recon_loss_method=recon_loss_method)  # Train model on train dataset
            testl = test(epoch, self.model, self.test_loader, recon_loss_method=recon_loss_method)

            if self.plot_loss:  # log train and test losses for dynamic plot
                logs = {}
                logs['' + 'ELBO'] = tl
                logs['val_' + 'ELBO'] = testl
                liveloss.update(logs)
                liveloss.draw()

    def run_vae(self):
        """
        Run the full data set through the trained vae model.

        Returns
        -------
        Modifies the zs attribute, an array of shape (number_traces, latent_space)
        """
        _, zs = forward_all(self.model, self.all_loader)
        return zs.numpy()

    def reduce(self, epochs=5, hidden_size=8, lr=1e-2, recon_loss_method='mse', plot_loss=True):
        """
        Controller function for the vae model.

        Parameters
        ----------
        epochs : int
            Number of epochs to run vae model.
        hidden_size : int
            Size of the vae model latent space representation.
        lr : float
            Learning rate for vae model training.
        recon_loss_method : str
            Method for reconstruction loss calculation
        plot_loss : bool
            Control on whether to plot the loss on vae training.

        Returns
        -------
        Modifies embedding attribute via generation of the low dimensional representation.

        """
        if hidden_size < 2:
            raise Exception('Please use hidden size > 1')

        self.plot_loss = plot_loss  # define whether to plot training losses or not
        self.create_dataloader()

        if not self.loaded_model:
            self.train_vae(epochs=epochs, hidden_size=hidden_size, lr=lr, recon_loss_method=recon_loss_method)

        self.embedding = self.run_vae()  # arbitrary dimension output from VAE


class BVaeModel(ModelAgent):
    """
    Runs the VAE model to reduce the seismic data to an arbitrary sized dimension, visualised in 2 via UMAP.
    """
    def __init__(self, data):
        super().__init__(data)
        self.name = 'beta_VAE'

    def create_dataloader(self, batch_size=32):
        """
        Create pytorch data loaders for use in vae training, testing and running.

        Parameters
        ----------
        batch_size : int
            Size of data loader batches.

        Returns
        -------
        Modifies object data loader attributes.

        """
        # create torch tensor
        assert self.input.shape[1] == 2, 'Expected a three dimensional input with 2 channels'
        X = torch.from_numpy(self.input).float()

        # Create a stacked representation and a zero tensor so we can use the standard Pytorch TensorDataset
        y = torch.from_numpy(np.zeros((X.shape[0], 1))).float()

        split = ShuffleSplit(n_splits=1, test_size=0.5)
        for train_index, test_index in split.split(X):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

        train_dset = TensorDataset(X_train, y_train)
        test_dset = TensorDataset(X_test, y_test)
        all_dset = TensorDataset(X, y)

        kwargs = {'num_workers': 1, 'pin_memory': True}
        self.train_loader = torch.utils.data.DataLoader(train_dset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        **kwargs)
        self.test_loader = torch.utils.data.DataLoader(test_dset,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       **kwargs)
        self.all_loader = torch.utils.data.DataLoader(all_dset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      **kwargs)

    def train_vae(self, epochs=5, hidden_size=8, lr=1e-2, beta=10, recon_loss_method='mse'):
        """
        Handles the training of the vae model.

        Parameters
        ----------
        epochs : int
            Number of complete passes over the whole training set.
        hidden_size : int
            Size of the latent space of the vae.
        lr : float
            Learning rate for the vae model training.
        beta : float
            Beta value adjusts the weight of importance in KLD term in loss function
        recon_loss_method : str
            Method for reconstruction loss calculation
        Returns
        -------
        None

        """
        set_seed(42)  # Set the random seed
        self.model = VAE(hidden_size,
                         self.input.shape)  # Initialize the model

        # Create a gradient descent optimizer
        optimizer = optim.Adam(self.model.parameters(),
                               lr=lr,
                               betas=(0.9, 0.999))

        if self.plot_loss:
            liveloss = PlotLosses()
            liveloss.skip_first = 0
            liveloss.figsize = (16, 10)

        # Start training loop
        for epoch in range(1, epochs + 1):
            tl = train(epoch,
                       self.model,
                       optimizer,
                       self.train_loader,
                       beta=beta,
                       recon_loss_method=recon_loss_method)  # Train model on train dataset
            testl = test(epoch, self.model, self.test_loader,
                         beta=beta,
                         recon_loss_method=recon_loss_method)  # Validate model on test dataset

            if self.plot_loss:
                logs = {}
                logs['' + 'ELBO'] = tl
                logs['val_' + 'ELBO'] = testl
                liveloss.update(logs)
                liveloss.draw()

    def run_vae(self):
        """
        Run the full data set through the trained vae model.

        Returns
        -------
        Modifies the zs attribute, an array of shape (number_traces, latent_space)
        """
        _, zs = forward_all(self.model, self.all_loader)
        return zs.numpy()

    def reduce(self, epochs=5, hidden_size=8, lr=1e-2, beta=5, recon_loss_method='mse', plot_loss=True):
        """
        Controller function for the vae model.

        Parameters
        ----------
        epochs : int
            Number of epochs to run vae model.
        hidden_size : int
            Size of the vae model latent space representation.
        lr : float
            Learning rate for vae model training.
        beta : float
            Beta value adjusts the weight of importance in KLD term in loss function
        recon_loss_method : str
            Method for reconstruction loss calculation
        plot_loss : bool
            Control on whether to plot the loss on vae training.

        Returns
        -------
        Modifies embedding attribute via generation of the low dimensional representation.

        """
        if hidden_size < 2:
            raise Exception('Please use hidden size > 1')

        self.plot_loss = plot_loss  # define whether to plot training losses or not
        self.create_dataloader()  # create datasets

        if not self.loaded_model:
            self.train_vae(epochs=epochs, hidden_size=hidden_size, lr=lr,
                           beta=beta, recon_loss_method=recon_loss_method)

        self.embedding = self.run_vae()  # arbitrary dimension output from bVAE


def plot_agent(model, attr='FF'):
    """
    Plots a low dimensional representation of seismic data found via model analysis.

    Parameters
    ----------
    model : object
        ModelAgent instance that has been run via the reduce method of the daughter class.
        This ensures the model has been run and the embedding attribute has been created.
    attr : str
        Represents the key for the attributes dictionary. Controls the attribute to be represented via a colour-scale
        in the resulting plot.

    Returns
    -------
    pyplot axes object.

    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set(xlabel='Latent Variable 1',
           ylabel='Latent Variable 2',
           title='Model used: {}, Trace Attribute: {}'.format(
               model.name, attr),
           aspect='equal')
    scatter = ax.scatter(model.two_dimensions[:, 0],
                         model.two_dimensions[:, 1],
                         s=1.0,
                         c=model.attributes[attr])
    cbar = plt.colorbar(scatter, shrink=0.7, orientation='vertical')
    cbar.set_label(label=attr, rotation=90, labelpad=10)
    plt.show()
    return scatter
