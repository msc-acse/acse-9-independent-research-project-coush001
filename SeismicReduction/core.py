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

# File load and save imports
from .utils import *


# live loss plots
from livelossplot import PlotLosses


### END OF IMPORTS

# processor
class DataHolder:
    def __init__(self, field_name, inlines, xlines):
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
        self.near, twt = load_seismic(fname, inlines=self.inlines, xlines=self.xlines)
        self.twt = twt

    def add_far(self, fname):
        self.far, twt = load_seismic(fname, inlines=self.inlines, xlines=self.xlines)
        assert (self.twt == twt).all, "This twt does not match the twt from the previous segy"

    def add_horizon(self, fname):
        self.horizon = interpolate_horizon(load_horizon(fname, inlines=self.inlines, xlines=self.xlines))

    def add_well(self, well_id, well_i, well_x):
        self.wells[well_id] = [well_i, well_x]


class Processor:
    def __init__(self, Data):
        self.raw = [Data.far, Data.near]
        self.twt = Data.twt
        self.out = None

        # attributes
        self.attributes = {'horizon_raw': Data.horizon}

    def flatten(self, data, top_add=12, below_add=52):
        out = []
        horizon = self.attributes['horizon_raw']

        # input data = [near(twt, x1, x2),far(twt, x1, x2)]
        for amplitude in data:
            # create output trace shape for each set in shape: (twt, x1, x2)
            traces = np.zeros((top_add + below_add, horizon.shape[0], horizon.shape[1]))
            for i in range(horizon.shape[0]):
                #  find the corresponding index of the horizon in amplitude twt 'domain'
                hrz_idx = [np.abs(self.twt - val).argmin() for val in horizon[i, :]]
                for j in range(horizon.shape[1]):
                    # place the twt's from above_below horizon into 3rd index
                    traces[:, i, j] = amplitude[hrz_idx[j] - top_add:hrz_idx[j] + below_add, i, j]
            out.append(traces)

        return out  # list of far and near, flattened amplitudes shape (twt, x1,x2)

    def roll_axis(self, data):
        # input should be (twt,x1,x2)
        for i in range(len(data)):
            data[i] = np.transpose(data[i], (1, 2, 0))
        # output (x1,x2,twt)
        return data

    def normalise(self, data):
        well_i = 38
        well_x = 138
        out = []
        for i in data:
            well_variance = np.mean(np.std(i[well_i - 2:well_i + 1, well_x - 2:well_x + 1], 2))
            i /= well_variance
            out.append(i)

        return out

    def to_2d(self, data):
        return [i.reshape(-1, data[0].shape[-1]) for i in data]

    def stack_traces(self, data):
        return np.stack([data[0], data[1]], axis=1)

    def run_AVO(self):
        #         print(self.out[0].shape, self.out[1].shape)
        x_avo = self.out[1]
        y_avo = self.out[0] - self.out[1]

        lin_reg = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
        lin_reg.fit(x_avo.reshape(-1, 1), y_avo.reshape(-1, 1))

        self.attributes['FF'] = y_avo - lin_reg.coef_ * x_avo

    def condition_attributes(self):
        #  flatten horizon array
        horizon = self.attributes['horizon_raw']
        self.attributes['horizon'] = horizon.reshape(horizon.shape[0] * horizon.shape[1])

        #  condense fluid factor to min of array
        self.attributes['FF'] = np.min(self.attributes['FF'], 1)

    def __call__(self, flatten=False, normalise=False):
        self.out = copy.copy(self.raw)

        if flatten[0]:
            self.out = self.flatten(self.out, flatten[1], flatten[2])

        self.out = self.roll_axis(self.out)

        if normalise:
            self.out = self.normalise(self.out)

        # flatten to 2d (traces, amplitudes)
        self.out = self.to_2d(self.out)

        # Find fluid factor, add to attributes
        self.FF = self.run_AVO()

        # condition attributes to 1d arrays
        self.condition_attributes()

        #  Stack the traces for output
        self.out = self.stack_traces(self.out)
        print('Processor has made an output with shape: ', self.out.shape)

        return [self.out, self.attributes]

# model
class ModelAgent:
    def __init__(self, data):
        self.input = data[0]
        self.attributes = data[1]
        self.embedding = None
        self.input_dimension = self.input.shape[-1]

        # for logging
        #         today = datetime.date.today()
        #         self.path = './runs/{}'.format(today)
        #         if not os.path.exists(self.path):
        #             os.mkdir(self.path)
        #             print("Directory " , self.path ,  " For Logs Created ")

        print("ModelAgent initialised")


class UMAP(ModelAgent):
    def __init__(self, data):
        super().__init__(data)
        self.name = 'UMAP'

    def concat(self):
        #         self.input = np.concatenate([self.input[0],self.input[1]], 1)

        self.input = self.input.reshape(-1, 2 * self.input_dimension)
        print('to enter UMAP:', self.input.shape)

    def reduce(self, n_neighbors=50, min_dist=0.001):
        # Directory for logging runs
        #         now = datetime.datetime.now().strftime("%I-%M-%S-%p")
        #         self.path = self.path + '/{}/'.format(now, self.name)

        self.concat()  # concat near_far into 1 dim

        embedding = umap.UMAP(n_neighbors=n_neighbors,
                              min_dist=min_dist,
                              metric='correlation',
                              verbose=False,
                              random_state=42).fit_transform(self.input)

        self.embedding = embedding

        print("UMAP 2-D representation complete")


class VAE_model(ModelAgent):
    def __init__(self, data):
        super().__init__(data)
        self.name = 'VAE'

    def create_dataloader(self, batch_size=32):
        # create torch tensor
        assert self.input.shape[1] == 2, 'expecting a 3D input'
        X = torch.from_numpy(self.input).float()
        # split the concatenated input back into two arrays
        #         X = torch.from_numpy(np.stack(np.split(self.input, 2, axis=1), 1)).float()

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
        self.train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=False, **kwargs)
        self.all_loader = torch.utils.data.DataLoader(all_dset, batch_size=batch_size, shuffle=False, **kwargs)

    def train_vae(self, cuda=False, epochs=5, hidden_size=8, lr=1e-2):
        set_seed(42)  # Set the random seed
        self.model = VAE(hidden_size, self.input.shape)  # Inititalize the model

        # Create a gradient descent optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))

        liveloss = PlotLosses()
        liveloss.skip_first = 0
        liveloss.figsize = (16, 10)  # , fig_path=self.path
        liveloss.fig_path = './'

        # Start training loop
        for epoch in range(1, epochs + 1):
            tl = train(epoch, self.model, optimizer, self.train_loader, cuda=False)  # Train model on train dataset
            testl = test(epoch, self.model, self.test_loader, cuda=False)  # Validate model on test dataset
            #             %matplotlib inline
            logs = {}
            logs['' + 'ELBO'] = tl
            logs['val_' + 'ELBO'] = testl
            liveloss.update(logs)
            liveloss.draw()

    def run_vae(self):
        _, self.zs = forward_all(self.model, self.all_loader, cuda=False)

    def vae_umap(self, umap_neighbours=50, umap_dist=0.001):
        print('\nVAE->UMAP representation initialised\n')
        transformer = umap.UMAP(n_neighbors=umap_neighbours,
                                min_dist=umap_dist,
                                metric='correlation', verbose=True).fit(self.zs.numpy())
        embedding = transformer.transform(self.zs.numpy())
        print("\n\nVAE -> 2-D UMAP representation complete\n")
        return embedding

    def reduce(self, epochs, hidden_size, lr, umap_neighbours, umap_dist):
        if hidden_size < 2: raise Exception('Please use hidden size > 1')

        # Directory for logging runs
        #         now = datetime.datetime.now().strftime("%I-%M-%S-%p")
        #         self.path = self.path + '/{}/'.format(now, self.name)

        # TODO create text file detailing all hyper parameters.

        #         if not os.path.exists(self.path):
        #             os.mkdir(self.path)
        #             print("Directory " , self.path ,  " For Logs Created ")

        self.create_dataloader()
        self.train_vae(epochs=epochs, hidden_size=hidden_size, lr=lr)
        self.run_vae()

        # Find 2-D embedding
        if hidden_size > 2:
            self.embedding = self.vae_umap(umap_dist=umap_dist, umap_neighbours=umap_neighbours)
        elif hidden_size == 2:
            self.embedding = self.zs.numpy()

#plot
def PlotAgent(model, attr='FF'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set(xlabel='Latent Variable 1', ylabel='Latent Variable 2',
           title='Model used: {}, Trace Attribute: {}'.format(model.name, attr),
           aspect='equal')
    s = ax.scatter(model.embedding[:, 0], model.embedding[:, 1], s=1.0, c=model.attributes[attr])
    c = plt.colorbar(s, shrink=0.7, orientation='vertical')
    c.set_label(label=attr, rotation=90, labelpad=10)
    plt.show()

