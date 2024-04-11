import numpy as np
import pandas as pd

og_data = pd.read_csv("dresses10_urls.csv")
og_data = og_data.loc[:, 'E1':].values
og_mean = np.mean(og_data)
og_std = np.std(og_data)

data_files = [
    "dresses10_url-PCA-dimensions5.csv",
    "dresses10_url-UMAP-dimensions5.csv",
    "vgg_pca10_dresses.csv",
    "vgg_pca20_dresses.csv",
]

for file in data_files:
    data = pd.read_csv(file)
    rescaled = data.loc[:, 'E1':].values
    rescaled = (rescaled - rescaled.mean()) / rescaled.std()
    rescaled = rescaled * og_std + og_mean
    data.loc[:, 'E1':] = rescaled
    data.to_csv("scaled_" + file, index=False)