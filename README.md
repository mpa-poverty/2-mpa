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
├── data                 # ignored path, default path to store output and input data  
├── data_handlers             
│   ├── datasets         # custom dataset backbones for torchgeo, csv  
│   └── samplers         # custom samplers for torchgeo               
├──  models                  
│   ├── resnet18.py      # MS Only `(Yeh&al., 2020)`  
│   ├── dual_branch.py   # MS + NL `(Yeh&al., 2020)`  
│   └── checkpoints      
└── utils                
```

### Notes
`03-20`: not using `torchgeo` that requires underlying lightning workflow as pre-trained networks provider for now. Instead, we downloaded the checkpoint files directly from SSL4EO [repository](https://github.com/zhu-xlab/SSL4EO-S12). 
