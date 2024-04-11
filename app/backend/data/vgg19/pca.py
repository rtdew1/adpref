from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

dresses_10 = pd.read_csv("dresses10_urls.csv")

final_fc_layers = np.load("final_fc_layer.npz")
final_fc_layers = final_fc_layers["arr_0"]


# Creating and exporting PCA
def export_pca_urls(n_components, input_data):
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(input_data)

    df = pd.DataFrame(transformed, columns=["E" + str(i) for i in range(1, n_components + 1)])
    df.insert(0, "URL", dresses_10["URL"])
    df.insert(0, "Item_ID", dresses_10["Item_ID"])
    df.to_csv("dresses_pca" + str(n_components) + ".csv", index=False)


export_pca_urls(10, final_fc_layers)
export_pca_urls(20, final_fc_layers)
