# 2-MPA : MAPPING POVERTY in AFRICA w.r.t. MARINE PROTECTED AREAS

### REQUIREMENTS
This code was tested on a system with the following specifications:  
- OS:  
- CPU:  
- GPU:  
- RAM:  

This code requires `Python >= ` with `Pytorch v.`, `Torchvision >=` and `Torchgeo >=`.

### CONTENT

```
├── configs              # config `json` files for each model  
├── data                 # ignored path, default path to store data\
│   └── ...dataset.csv   # available datasets in csv format 
├── data_handlers        # custom Dataset / DataLoader for torch            
├── models                 
│   ├── resnet18.py      # MS Only `(Yeh&al., 2020)`  
│   └── checkpoints      
├── preprocessing        # notebooks to reproduce downloading and prep'ing the data   
├── utils 
├── test.py   
└── train.py                
```
