# SEISMIC REDUCTION 
## Unsupervised Machine Learning: An Application to Seismic Amplitude vs Offset Interpretation ##

This project delivers a set of tools to run unsupervised machine learning on seismic data sets. The aim is to enable the  efficient experimentation of an array of models and the range of paramaters therein. Specifically the tools allow for efficient recognition of clustering of low fluid-factor anomalies derived from AVO analysis.

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

### Example data:
* The example data set used in the project can be acquired from this link:<br>
[Example dataset[zip, 205mb]](http://pangea.stanford.edu/departments/geophysics/dropbox/SRB/public/data/qsiProject_data.zip) *- curteousy of Stanford University*
- Only the following files need to be extracted:
> '3d_farstack.sgy', '3d_nearstack.sgy', 'Top_Heimdal_sebset.txt'


### Usage:
There are two ways to utilise the software, each with different merits:
1. **Direct python scipting:** Access the python package tools directly.
    * Access to model load/save capabilities for key models, or models run for large number of epochs.
    * Ability to nest runs in a list to efficiently explore a consistent range of parameters.
    * Ability to save the scripts of particularly useful analysis
    * Ability to utilise distributed computing for large jobs without dependance on a notebook.
    
2. **Jupyter Notebook GUI:** Interface with very a straighforward self-explanatory graphical interface.
    * Easy access to analysis without need of coding experience.
    * Quick and easy to run an analysis with a few button clicks without having to write code.
    * Pickle checkpointing allows for quick runs without having to repeat costly dataloading (similar benefits with Jupyter cells)
  
---

## 1. Direct python scripting:

The tool is delivered via a series of classes delivering the following workflow:
1. Imports
2. Data Loading
3. Data processing
4. Model analysis
  1. Model embedding to a chosen dimension *Example: PCA, VAE..*
  2. Umap embedding to two dimensions
5. Visualisation
  1. Choice of attribute (colour scale) overlay *Example: fluid factor, horizon depth*


## Work Flow:

### 1.1 Importing
* Run in the standard way. Can also choose to import individual classes but there isn't many so the namespace will not be swamped.
```python
from SeismicReduction import *
```
### 1.2 Data loading
* Data loading is done via the **DataHolder** class. <br>
* Initialisation takes three parameters:
    1. Dataset name : self explanatory
    2. inline range : in the form [start, stop, step]
    3. inline range : in the form [start, stop, step] <br>
*if using test dataset use the below ranges, if using new data check the info documentation for this*
```python
# init
dataholder = DataHolder("Glitne", [1300, 1502, 2], [1500, 2002, 2])
```
* Loading the near and far offset amplitudes files is self explanatory, use the relative pathname of the files. <br>
* *files **must** be in .sgy format*
```python
# add near and far offset amplitudes
dataholder.add_near('./data/3d_nearstack.sgy');
dataholder.add_far('./data/3d_farstack.sgy');
```
* Loading the horizon, information must be in .txt with columns: inline, crossline, twt 
```python
# add the horizon depth
dataholder.add_horizon('./data/Top_Heimdal_subset.txt')
```
### 1.3 Data processing
* Uses class **Processor**. A processor only needs to be initialised **once** per dataset, the parameter is the DataHolder object. <br>
```python
# Create a processor object for the data
processor = Processor(dataholder)
```
* An input is generated from the object __\_\_call\_\___ with the following parameters:
    1. flatten : list with three elements [bool, int: above add, int: below add]
        * element one chooses whether to run horizon flattening in the dataset
        * above and below add choose how many amplitudes either side of the horizon to extract
    2. crop : list with three elements [bool, int: above index, int: below index]
        * element one chooses whether to run cropping on the dataset
        * above and below index choose the extents of the seismic window to be extracted
    3. normalise : bool
        * chooses whether to normalise the data or not
* **Note:** If both flattening and cropping are true, only flattening will occur. 
```python
# Generate an output, first param specifies flattening procedure, second specifies normalisation
input = processor(flatten=[True, 12, 52], crop=[False, 0, 232], normalise=True)
```
### 1.4 Model analysis
* The available unsuperverised machine learning techniques are available in the following classes:
    1. Principal Component Analysis: **PcaModel**
    2. Uniform Manifold Approximation: **UmapModel**
    3. Variational Auto Encoder: **VaeModel**
    4. Beta-Varational Auto Encoder: **BVaeModel**

* Each model must be initialised with an input generated from the processor object.  
```python
# initialise a VAE model on the input
vae = VaeModel(input2)
```
* For every model the next step is to run the **.reduce()** method.
* Depending on the model, the parameter options vary.
```python
# reduce to lower dimension
vae.reduce(epochs=5, hidden_size=2, lr=1e-2, umap_neighbours=50, umap_dist=0.001, plot_loss=True)
```
### 1.4.2 Two dimension UMAP embedding
* Regardless of the model, after **.reduce()**, **.to_2d()** must be run to convert to a 2d representation of the embedding via umap. If already reduced to 2d via the model this method must still be run to configure internal data.
* Parameters:
   1. umap_neighbours : the n_neighbours parameter used by the umap algorithm
   2. umap_dist : the min_dist parameter used by the umap algorithm
```python
# reduce to 2d with umap
vae.to_2d(umap_neighbours=50, umap_dist=0.02)
```

### 1.5 Visualisation
* Visualisation is run by a standalone function.

* Parameters
    1. model : a model object initialised, embedded,and converted to 2d
    2. attr : to plot fluid factor use "FF", for horizon depth use "horizon"
    
```python
# Plot the vae representation with the AVO fluid factor attribute overlain
PlotAgent(model=vae, attr="FF")
```
### result
(https://github.com/msc-acse/acse-9-independent-research-project-coush001/images/test)
---

## 2. Notebook GUI:
A jupyter notebook GUI that captures all of the tools functionality without need to edit code has been created and is available for download from this repo *'GUI_tool.ipynb'*.

The tool is delivered via jupyter widgets and follows the same workflow as the api tools:
- Data Loading->Data processing->Model analysis->Visualisation.

Via the use of pickling (the python module that allows for saving and loading of python objects) the analysis can be run in a segmented fashion. For example it is only necessary to load the data once, this can then be processed in a number of ways and in turn a number of different models can be run on one set of processed data.

---

### Documentation
- Detailed documentation of all python functions and classess.

## Testing
Continuous integration is deployed using the travis framework.

### Build Status
[![Build Status](https://travis-ci.com/msc-acse/acse-9-independent-research-project-coush001.svg?branch=master)](https://travis-ci.com/msc-acse/acse-9-independent-research-project-coush001)

### Integration testing
Integration testing involves running an anaylis using the current state of software and compares the final output against a documented and verified base case result. These tests ensure the full analysis pipeline is working as expected and is creating the expected outputs. Due to the stochastic nature of the algorithms, these are standardised with random seed setting.

Currently integration tests have to be run locally as there are some undiagnosed errors that appear when run with travis.

### Unit testing
Unit testing is employed to ensure the functions and methods of the software are delivering the expected outputs.
These are particularly focussed on running checks on the shapes of inputs and outputs from the functions and methods to ensure the expected data is being generated in the correct way.

---

## Built with:
python 3

## License
[MIT license](LICENSE)
