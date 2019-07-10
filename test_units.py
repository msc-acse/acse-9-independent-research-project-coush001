import pytest
from SeismicReduction import VaeModel, VAE, UmapModel, DataHolder, Processor, PcaModel, BVaeModel
import torch


# loading data : very costly do once for all integration tests!
dataholder = DataHolder("Glitne", [1300, 1502, 2], [1500, 2002, 2])
dataholder.add_near('./data/3d_nearstack.sgy');
dataholder.add_far('./data/3d_farstack.sgy');
dataholder.add_horizon('./data/Top_Heimdal_subset.txt')
dataholder.add_well('well_1', 36, 276 // 2)

# processor
processor = Processor(dataholder)
not_flat_input = processor([False, 0, 232], True)
flat_input = processor([True, 12, 12], True)

def test_utils_VAE():
    # testing VAE architecture input matches output
    x = torch.randn((1, 2, 64))
    model = VAE(8, x.shape)
    y = model(x)
    assert(x.shape == y[0].shape), 'VAE input does not match output'

def test_data_holder_load_correct_shape():
    assert(dataholder.near.shape == dataholder.far.shape), \
        'For test case, load functions have loaded files incorrectly'
    assert(dataholder.near.shape[1:] == dataholder.horizon.shape[:]), \
        'Twt axis not in the expected 0 index of amplitudes array'
    return 1

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
        'Shape of attribute should be irrespective of input routine'
    return 1


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