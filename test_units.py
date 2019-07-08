import pytest
from SeismicReduction import VAE_model, VAE
import torch

def test_ok():
    print("ok")

def test_utils_VAE():
    # testing VAE architecture input matches output
    x = torch.randn((1, 2, 64))
    model = VAE(8, x.shape)
    y = model(x)
    assert(x.shape == y[0].shape), 'VAE input does not match output'

def test_data_holder():
    return 1

def test_Processor():
    return 1

def test_Model_agent():
    return 1

def test_VAE_model():
    return 1

def test_UMAP():
    return 1