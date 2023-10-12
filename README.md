# Improving Poverty Estimation

### CONTENT

```
├── configs              # config `json` files for each model and for some tests
├── data                 # ignored path, default path to store data\
│   ├── additional_data  # time-series (climatic and conflict data)
│   ├── landsat_7_less   # landsat multispectral + nighlights tif images
│   └── ...dataset.csv   # available datasets in csv format
│ 
├── datasets             # custom Datasets for torch
├── figures              # output results figures saved in .png
├── models                 
│   ├── build_models.py  # models constructors
│   └── checkpoints
├── plot_results         # notebook to create test dataset, compute R2 and plot the results
├── postprocessing       # notebook to handle checkpoints after a training is launched 
├── preprocessing        # notebooks to reproduce downloading and preparing the data
├── results              # test dataset with updated predictions from models
├── testing              # test loop
├── training             # train loops (per model)
├── grid_search.py       # script to trained models
├── use_model.py         # script to use the model to predict poverty on new clusters (using existing checkpoints)
└── utils
```
### REQUIREMENTS
OS: Ubuntu 18.04 LTS  
PY: Python 3.10  
GPU: Nvidia Quadro RTX 6000

##### Virtual Env (w.i.p.)
- Requires installation of `python3.10` 
- In the 2-mpa directory:  
```pip install virtualenv```  
```virtualenv venv```  
```source venv/bin/activate```  
```pip install -r requirements.txt```

### SCRIPT LAUNCH
To launch model training or usage, write the corresponding command in the terminal.  

#### To train a model:
to train a model (single branch of 2/3-branches fusion):

```python3 ./grid_search.py configs/model_gs.json model_type```

###### model type: 
ms: CNN learning on multi-spectral Landsat images
vit_ms: ViT learning on multi-spectral Landsat images
nl: CNN learning on night lights
ts: FCN learning on additional data (socio-climatic)
msnl: 2-branches model, ms+nl
msnlt: 3-branches model, ms+nl+ts
vit_msnlt: 3-branches model, vit_ms+nl+ts


#### To use a model:
To use a model to predict poverty on new clusters (using existing checkpoints):

```python3 ./use_model.py configs/model_gs.json model_type csv_path image_path series_path```

