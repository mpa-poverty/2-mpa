# Improving Poverty Estimation

##### Virtual Env (w.i.p.)
- Necessite une installation de `python3.10` 
- Dans le repertoire 2-mpa :  
```pip install virtualenv```  
```virtualenv venv```  
```source venv/bin/activate```  
```pip install -r requirements.txt```
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
├── preprocessing        # notebooks to reproduce downloading and prep'ing the data
├── results              # dataset with updated predictions from models
├── testing              # test loop
├── training             # train loops (per model)
├── predict.py           # script to predict poverty from new data with trained models
└── utils
```
