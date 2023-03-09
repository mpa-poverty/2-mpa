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
├── configs                  # config `json` files for each model  
├── data                     # ignored path, default path to store output and input data  
├──  models                  
│   ├── resnet18.py           # MS Only `(Yeh&al., 2020)`  
│   ├── dual_branch.py        # MS + NL `(Yeh&al., 2020)`  
│   └── checkpoints           # models checkpoints at training times  
│  
├──  tile_with_vec            # package to handle working with both csv and raster data  
│   ├── datasets.py           # custom dataset backbones for torchgeo   
│   └── utils                
│       ├── csv_utils.py      # functions to handle csv data  
│       └── torchgeo_utils.py # custom torchgeo classes and helper functions  
└── utils                   
    ├── transfer_learning.py  # functions to tweak pre-trained networks  
    └── utils.py              # misc. helper functions
```
