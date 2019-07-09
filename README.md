# SEISMIC REDUCTION 
### Unsupervised Machine Learning: An Application to Seismic Amplitude vs Offset Interpretation ###

The aim of this project is to deliver a set of tools to enable the efficient exploration of the potential use of unsupervised machine learning techniques for the analysis of seismic data sets specifically the recognition of clustering of low fluid-factor anomalies derived from AVO analysis.

---

## Getting started:

### Installation:
- The package is hosted at https://pypi.org/project/SeismicReduction/
- Install with the following in terminal:
```bash
pip install SeismicReduction
```
- The raw package tools are now available in any python script or jupyter notebook using standard import:
```python
import SeismicReduction
```

### Example Usage:

#### Direct python scripting:

The tool is delivered via a series of classes delivering the following workflow:
- Data Loading
- Data processing
- Model analysis
- Visualisation

These can be run in the following use case:

##### Imports
```python
from SeismicReduction import DataHolder, Processor, UMAP, VAE_model, set_seed, PlotAgent
```
##### Data loading
```python
# initiate data holder with arbritary name, inline range, and cross-line range
dataholder = DataHolder("Glitne", [1300, 1502, 2], [1500, 2002, 2])

# add near and far offset amplitudes
dataholder.add_near('./data/3d_nearstack.sgy');
dataholder.add_far('./data/3d_farstack.sgy');

# add the horizon depth
dataholder.add_horizon('./data/Top_Heimdal_subset.txt')
```
##### Data processing
```python
# Create a processor object for the data
processor = Processor(dataholder)

# Generate an output, first param specifies flattening procedure, second specifies normalisation
input2 = processor([True, 12, 52], normalise=True)
```
##### Unsupervised model analysis
```python
# run a VAE model on the input
vae_1 = VAE_model(input2)
vae_1.reduce(epochs=5, hidden_size=2, lr=1e-2, umap_neighbours=50, umap_dist=0.001, plot_loss=True)

# run a UMAP model on the input
umap = UMAP(Input1)
umap.reduce(n_neighbors=50, min_dist=0.001)
```
##### Visualisation
```python
# Plot the vae_1 representation with the AVO fluid factor attribute overlain
PlotAgent(vae_1, "FF")

# Plot the UMAP representation witht the horizon depth plotted as overlain attribute
PlotAgent(umap)
```

#### Notebook GUI:
A jupyter notebook GUI that captures all of the tools functionality without need to edit code has been created and is available for download from this repo *'GUI_tool.ipynb'*.

The tool is delivered via jupyter widgets and follows the same workflow as the api tools:
- Data Loading->Data processing->Model analysis->Visualisation.

Via the use of pickling (the python module that allows for saving and loading of python objects) the analysis can be run in a segmented fashion. For example it is only necessary to load the data once, this can then be processed in a number of ways and in turn a number of different models can be run on one set of processed data.

### Documentation
- Detailed documentation of all python functions and classess.

---

## Testing
All testing will be included in the travis testing routine for continuous integration of features

### Build Status
[![Build Status](https://travis-ci.com/msc-acse/acse-9-independent-research-project-coush001.svg?branch=master)](https://travis-ci.com/msc-acse/acse-9-independent-research-project-coush001)

### Integration testing
Base cases wich have been verified to give the expected result have been logged and will be used in the integration tests to ensure the full analysis pipeline is working as expected

### Unit testing
Unit testing is employed to ensure the functions and methods of the software are delivering the expected outputs

---

## Built with:
python 3

## Contributing

## Author
Hugo Coussens

## License
[MIT license](LICENSE)
