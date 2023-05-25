# 2-MPA : MAPPING POVERTY in AFRICA w.r.t. MARINE PROTECTED AREAS

### UPDATES
##### Clusters & dataset w/ clusters
- clusters a telecharger : `data/dhsnl_folds.pkl`
- dataset avec clusters assignes : `data/dataset_with_clusters.csv`
##### Virtual Env 
- Necessite une installation de `python3.10` 
- Dans le repertoire 2-mpa :  
```pip install virtualenv```  
```virtualenv venv```  
```source venv/bin/activate```  
```pip install -r requirements.txt```
##### Notes sur le preprocessing / earth-engine
- Dans le fichier `preprocessing/ee_utils.py`, lignes 298-303 :  
```
# Collection d'images Landsat 8
self.l8 = self.init_coll('LANDSAT/LC08/C01/T1_SR').map(self.rename_l8).map(self.rescale_l8) 
# Collection d'images Landsat 7
# self.l7 = self.init_coll('LANDSAT/LE07/C01/T1_SR').map(self.rename_l57).map(self.rescale_l57)
# Collection d'images Landsat 5
# self.l5 = self.init_coll('LANDSAT/LT05/C01/T1_SR').map(self.rename_l57).map(self.rescale_l57)
# Ici on fusionne la collection landsat 7 :
# self.merged = self.l5.merge(self.l7).merge(self.l8).sort('system:time_start')
# La, non :
self.merged = self.l5.merge(self.l8).sort('system:time_start')
```
Dans le second cas, les bandes de Landsat 5 sont vides pour les annees <= 2012, ce qui est aberrant.  
Je n'ai pas reussi a resoudre le probleme.

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
