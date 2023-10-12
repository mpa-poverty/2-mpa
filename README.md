# Improving Poverty Estimation

### CONTENT

```
├── configs              # config `json` files for each model  
├── data                 # ignored path, default path to store data\
│   ├── additional_data  # time-series
│   ├── landsat_7_less   # landsat multispectral + nighlights tif images
│   └── ...dataset.csv   # available datasets in csv format
│ 
├── datasets             # custom Datasets for torch
├── figures          
├── models                 
│   ├── build_models.py  # models constructors
│   └── checkpoints
├── postprocessing       # notebook to handle checkpoints after a training is launched 
├── preprocessing        # notebooks to reproduce downloading and prep'ing the data
├── results              # dataset with updated predictions from models
├── testing              # test loop
├── training             # train loops (per model)
├── predict.py           # script to predict poverty from new data with trained models
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

```python3 ./grid_search.py configs/model_gs.json model_type```

```python3 ./use_model.py configs/model_gs.json model_type csv_path image_path series_path```

