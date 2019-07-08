import pytest
from SeismicReduction import DataHolder, Processor, UMAP, VAE_model, set_seed, PlotAgent
import numpy as np


# set initial random seed for standardised testing
set_seed(42)

# loading data : very costly do once for all integration tests!
dataholder = DataHolder("Glitne", [1300, 1502, 2], [1500, 2002, 2])
dataholder.add_near('./data/3d_nearstack.sgy');
dataholder.add_far('./data/3d_farstack.sgy');
dataholder.add_horizon('./data/Top_Heimdal_subset.txt')
dataholder.add_well('well_1', 36, 276//2)


def test_base_case_UMAP():
    assert (dataholder.field_name == "Glitne")

    # process data for input
    processor = Processor(dataholder)
    Input1 = processor([True, 12, 52], normalise=True)

    # run the UMAP model on the input
    UMAP_a = UMAP(Input1)
    UMAP_a.reduce(n_neighbors=50, min_dist=0.001)

    PlotAgent(UMAP_a)

    # load the documented base case result as np array
    UMAP_base_case = np.load('test_cases/Flat-top12,bottom52,UMAP-neigh50,dist0.001.npy')



    # Base Case testing:
    assert (UMAP_base_case.shape == UMAP_a.embedding.shape), 'output embedding array shape does not match' \
                                                             'base case embedding array shape'
    assert (type(UMAP_base_case.shape) == type(UMAP_a.embedding.shape)),'Embedding array is not in expected numpy array' \
                                                                        'format'

    assert((np.isclose(UMAP_base_case.shape,UMAP_a.embedding.shape, atol=0.01)).all()), 'The implementation does not match the base case for: \n' \
                                            'Flattened input : above 12, below 52, \n' \
                                            'UMAP reduction : Neighbours 50, min_dist: 0.001'
    return 1

def test_base_case_VAE():
    # process data for input
    processor = Processor(dataholder)
    Input2 = processor([True, 12, 52], normalise=True)

    # run the VAE model on input
    VAE_1 = VAE_model(Input2)
    VAE_1.reduce(epochs=5, hidden_size=2, lr=1e-2, umap_neighbours=50, umap_dist=0.001, plot_loss=True)

    # load base case
    VAE_base_case = np.load('test_cases/Flat-top12,bottom52,VAE-epochs5,latent2,lr0.01.npy')

    # Base case testing:
    assert (VAE_base_case.shape == VAE_1.embedding.shape), 'output embedding array shape does not match' \
                                                             'base case embedding array shape'
    assert (type(VAE_base_case.shape) == type(VAE_1.embedding.shape))

    assert((VAE_base_case == VAE_1.embedding).all()), 'The implementation does not match the base case for: \n' \
                                            'Flattened input : above 12, below 52, \n' \
                                            'VAE reduction : Epochs 5, latent dimension 2, learn rate 0.01'
    return 1