from SeismicReduction import *

set_seed(42) # set seed to standardise results

### Data loading:
dataholder = DataHolder("Glitne", [1300, 1502, 2], [1500, 2002, 2])
dataholder.add_near('./data/3d_nearstack.sgy');
dataholder.add_far('./data/3d_farstack.sgy');
dataholder.add_horizon('./data/Top_Heimdal_subset.txt')

### Processing:
processor = Processor(dataholder)
processed_data = processor(flatten=[True, 12, 52], crop=[False, 120, 200], normalise=True)

### Model analysis:

## PCA
pca = PcaModel(processed_data)
pca.reduce(2)
pca.to_2d()

## UMAP
umap = UmapModel(processed_data)
umap.reduce(umap_neighbours=50, umap_dist=0.01)

## vae
vae = VaeModel(processed_data)
vae.reduce(epochs=50, hidden_size=2, lr=0.0005, plot_loss=False)
vae.to_2d()

## bvae
bvae = BVaeModel(processed_data)
bvae.reduce(epochs=50, hidden_size=2, lr=0.0005, beta=7, plot_loss=False)
bvae.to_2d()

## Visualisation
plot_agent(vae, attr='FF', cmap='magma', vmin=-3 ,save_path=False)
plot_agent(bvae, attr='FF', cmap='hot',save_path=False)
plot_agent(vae, attr='FF', cmap='magma' ,save_path=False)
plot_agent(bvae, attr='FF', cmap='winter',save_path=False)