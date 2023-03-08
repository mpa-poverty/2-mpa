# 2-MPA : MAPPING POVERTY in AFRICA w.r.t MARINE PROTECTED AREAS

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
├── data                   
│   ├── data_loader          # custom dataloaders  
│   ├── datasets             # datasets used in the projects  
│   └── samplers             # custom samplers from torchgeo  
├──  models                  
│   ├── base.py              # model super class  
│   ├── resnet18.py          # baseline inspired from `(Yeh&al., 2020)`  
│   └── checkpoints          # models checkpoints at training times  
└── utils                   
    ├── transfer_learning.py # utils functions to tweak pre-trained networks  
    └── utils.py
```