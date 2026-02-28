# Quality_Inspection
Anomalies detection in industrial parts by deep learning


This classification project is based on kaggle dataset "casting product image data for quality inspection" (https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product?resource=download-directory&select=casting_data). To use the code, you need to download the folder casting_data from the kaggle and copy it in data folder.

The dataset contains total 7348 image data. These all are the size of (300*300) pixels grey-scaled images. In all images, augmentation already applied.
there are two categories:
- Defective parts
- Ok parts

Both train and test folder contains def_front and ok_front subfolders.

train: def_front have 3758 and ok_front have 2875 images
test: def_front have 453 and ok_front have 262 images

