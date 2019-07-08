from utils import *
from SeismicReduction import DataHolder, Processor, UMAP, VAE_model, set_seed, PlotAgent
import numpy as np
set_seed(42)

# loading data : very costly do once for all integration tests!
dataholder = DataHolder("Glitne", [1300, 1502, 2], [1500, 2002, 2])
dataholder.add_near('./data/3d_nearstack.sgy');
dataholder.add_far('./data/3d_farstack.sgy');
dataholder.add_horizon('./data/Top_Heimdal_subset.txt')
dataholder.add_well('well_1', 36, 276//2)

# process data for input
processor = Processor(dataholder)
Input1 = processor([True, 12, 52], normalise=True)

# run the UMAP model on input
umap = UMAP(Input1)
umap.reduce(n_neighbors=50, min_dist=0.001)

PlotAgent(umap)

np.save('test_cases/Flat-top12,bottom52,UMAP-neigh50,dist0.001.npy', umap.embedding)    # .npy extension is added if not given

# load base case
umap_base_case = np.load('test_cases/Flat-top12,bottom52,UMAP-neigh50,dist0.001.npy')

# test the implementation finds same result as basecase
assert ((umap_base_case == imap.embedding).all()), 'The implementation does not match the base case for: \n' \
                                                   'Umap n_neighbours 50, min_dist 0.001 \n'

if (VAE_base_case == VAE_1.embedding).all():
    print('yes')