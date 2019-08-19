import pytest
from SeismicReduction import VaeModel, VAE, UmapModel, DataHolder, Processor, PcaModel, load_seismic, \
    interpolate_horizon, load_horizon
import numpy as np
import torch


# loading data : very costly do once for all integration tests!
dataholder = DataHolder("Glitne", [1300, 1502, 2], [1500, 2002, 2])
dataholder.add_near('./data/3d_nearstack.sgy');
dataholder.add_far('./data/3d_farstack.sgy');
dataholder.add_horizon('./data/Top_Heimdal_subset.txt')

# processor
processor = Processor(dataholder)
not_flat_input = processor(crop=[True, 0, 232])
flat_input = processor(flatten=[True, 12, 12])

# 1. Test utils.py:
def test_load_seismic():
    amp, twt = load_seismic('./data/3d_nearstack.sgy', [1300, 1502, 2], [1500, 2002, 2])
    assert(amp.shape[0] == twt.shape[0]), 'amplitude depth dimension does not match twt array length'


def test_interpolate_horizon():
    interpolated = interpolate_horizon(load_horizon('./data/Top_Heimdal_subset.txt', [1300, 1502, 2], [1500, 2002, 2]))
    assert((interpolated != 0).all()), 'horizon has not been interpolated properly as zero values still present'


def test_VAE_output_correct_shape():
    # testing VAE architecture input matches output
    x = torch.randn((1, 2, 64))
    model = VAE(8, x.shape)
    y = model(x)
    assert(x.shape == y[0].shape), 'VAE input does not match output'


# 2. Test DataHolder
def test_data_holder_load_correct_shape():
    assert(dataholder.near.shape == dataholder.far.shape), \
        'For test case, load functions have loaded files incorrectly'
    assert(dataholder.near.shape[1:] == dataholder.horizon.shape[:]), \
        'Twt axis not in the expected 0 index of amplitudes array'
    return 1


# 3. Test Processor
def test_processor_flattened():
    above_add = 12
    below_add = 12
    flat_input = processor(flatten=[True, above_add, below_add])
    assert(flat_input[0].shape[-1] == above_add+below_add), 'Output generated does not return the expected shape output'


def test_processor_cropped():
    top_index = 10
    bottom_index = 20
    crop_input = processor(crop=[True, top_index, bottom_index])
    assert(crop_input[0].shape[-1] == bottom_index-top_index), 'Output generated does not return the expected shape output'


def test_processor_normalise():
    normalised = processor(normalise=True)
    assert(np.isclose(np.mean(normalised[0]), 0, atol=0.001)), 'Mean has not been normalised to 0'
    assert(np.isclose(np.std(normalised[0]), 1, atol=0.01)), 'standard deviation has not been normalised to 1'


def test_processor_collapse_dimension():
    first=10
    second=10
    stub = np.zeros((first, second,100))
    test = Processor.collapse_dimension(self=False, data=[stub])
    assert(test[0].shape[1] == first*second), 'Dimension collapse not returning expected shape'


def test_Processor_create_correct_shape_output():
    # 0 index is input
    # check the input shape is 3
    assert (len(not_flat_input[0].shape) == 3)
    assert (len(flat_input[0].shape) == 3)

    # check the input has length 2 in second dimension representing the near far channels
    assert (not_flat_input[0].shape[1] == 2), \
        'Expected 2 in input second dimension'
    assert (flat_input[0].shape[1] == 2), \
        'Expected 2 in input second dimension'

    # check the dimension of unflattened traces matches input
    assert(not_flat_input[0].shape[2] == 232), \
        'Unflattened processed traces should have same length as input traces.'
    # check flattened traces has expected dimension (below add + above add)
    assert(flat_input[0].shape[2] == 24), \
        'Flattened with 12 above+below should result with input length of 24.'

    # 1 index is 'attributes' dict, (example: fluid factor per trace)
    assert (flat_input[1]['FF'].shape[0] == flat_input[0].shape[0]), \
        'Shape of attribute does not match shape of data input'
    return 1


# 4. Test Models
def test_PCA_model():
    pca = PcaModel(flat_input)
    pca.reduce(2)
    assert(pca.embedding.shape[1] == 2), 'Resultant analysis dimension not equal to two'


def test_VAE_model():
    vae = VaeModel(flat_input)
    vae.reduce(epochs=5, hidden_size=2, lr=1e-2, plot_loss=False)
    assert(vae.embedding.shape[1] == 2), \
        'Resultant analysis dimension is not equal to two'
    return 1


def test_UMAP():
    umap = UmapModel(flat_input)
    umap.reduce()
    umap.to_2d(umap_neighbours=50, umap_dist=0.01)
    assert(umap.two_dimensions.shape[1] == 2), \
        'Resultant analysis dimension is not equal to two'
    return 1