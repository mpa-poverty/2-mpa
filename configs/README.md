# CONFIGS

For `TS` model 
```
   {
	"num_channels": 6,                         # total number of series
	"output_size": 1,                         
    "filter_size": 2,                          # filter_size << series length

	"lr": 1e-4,                                # lower if overfitting
	"batch_size": 256,                         # increase if overfitting (64, 128, 256, 512)
	"decay":1e-2,                              # increase if overfitting
	"n_epochs":100,
	
	"checkpoint_path": "models/checkpoints/ts",
	"result_path": "models/results/ts_"
}

``` 