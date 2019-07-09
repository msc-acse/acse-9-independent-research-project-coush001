from SeismicReduction import DataHolder, Processor, UMAP, VAE_model, set_seed
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
Input2 = processor([True, 12, 52], normalise=True)

# run the VAE model on input
VAE_1 = VAE_model(Input2)
VAE_1.reduce(epochs=5, hidden_size=2, lr=1e-2, umap_neighbours=50, umap_dist=0.001, plot_loss=True)

PlotAgent(VAE_1, 'FF')
PlotAgent(VAE_1, 'horizon')