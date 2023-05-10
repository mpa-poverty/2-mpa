# PREPROCESSING STEPS
Please execute the notebooks in naming order. We discuss each of them briefly below.
### Step 0: Download TFRecords
> Important Note: To set up the data from scratch, e.g. to use Google's Earth Engine (EE)services, there are a few steps to follow to set up the EE account and link its activity to your personal drive [here](ee_account_setting.md)
The first notebook downloads TFRecords using the helper functions provided by `ee_utils.py`.  
This code is taken from @Yeh_2020 repository and is manually updated to 2023 TensorFlow changes as well as earth-engine data availability.  
Please note that Google Earth Engine limits the number of simultaneous query to 3000. You might want to set up a checkpoint depending on your `CHUNK_SIZE`.
By executing this notebook completely, and by downloading the archives from your Drive, you should end up with : 
```
data/
    DHS_EXPORT_FOLDER/
        angola_2011_00.tfrecord.gz
        ...
        zimbabwe_2015_XX.tfrecord.gz   
```
### Step 1: Process TFRecords
We're still following Yeh's footsteps here. This notebook merges the chunks from previous steps into per-data point tfrecords.

### Step 2: Transform TFRecords
We then transform each TFRecord into a TIF file, which arguably takes more space but avoid having to rely on TensorFlow from now on, from a Pytorch workflow veiwpoint.  
We also compute the minimum and maximum values for each band across the whole dataset and normalize the data between 0 and 1.

### Step 3: Create Incountry Folds
This step is mandatory to perform spatially aware cross-validation to obtain a robust model.  
The code is taken from @Yeh_2020, still, and produces spatial folds from the dataset indices.
