# TerraFlood
An approach for automated SAR-based flood monitoring over Flanders.

This repository contains the source codes used in the following publication:

Landuyt et al., (in review). "Towards operational flood monitoring in Flanders using Sentinel-1". IEEE JSTARS.

The code itself can be referenced by the following DOI: [![DOI](https://zenodo.org/badge/346770017.svg)](https://zenodo.org/badge/latestdoi/346770017)

### Prerequisites
This repository is mainly built using the following packages:
```
python 3.6.8
scipy 1.3.1
scikit-learn 0.21.3
scikit-image 0.15.0
pandas 0.25.1
geopandas 0.5.1
shapely 1.6.4
rasterio 1.0.26
```

### Run the code
The main file to run the algorithm over a community in Flanders is TerraFlood_Community.py. This script requires 3 inputs:
- the name of the community
- the start date of the monitoring period, formated as YYYYmmdd
- the end date of the monitoring period, formated as YYYYmmdd

Flood maps are made for all Sentinel-1 images in the provided time range. The Sentinel-1 image pairs is saved as a .tif, while the resulting classifications are saved both as a .png and a .tif file. The following classes (with label) are present: 
- 0 = dry land (DL)
- 1 = permanent water (PW)
- 2 = open flooding (OF)
- 3 = probable flooding (PF)
- 4 = long-term flooding (LTF)
- 5 = flooded vegetation (FV)
- 6 = probably flooded forest (PFF)
- 7 = invisible forest (IF)

Alternatively, the algorithm can be ran over an area outside Flanders. To do so, L227-254 should be ran, and the following layers should be provided:
- VV band of flood image
- VH band of flood image
- VV band of ref. image
- VH band of ref. image
- VV increase band
- ratio increase band
- tree cover
- elevation
- permanent water layer (optional)
- pluvial flood probability (optional)
- fluvial flood probability (optional)
- pluvial flood depth (optional)
- fluvial flood depth (optional)
