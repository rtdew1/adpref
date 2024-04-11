from data_io.shared.util import unit_constrain
from config import DATA_FILE
import pandas as pd
import numpy as np

# Load data
all_items = pd.read_csv(DATA_FILE)

# Compute constrained z representations
z_unconstr_all = np.array(all_items.loc[:, "E1":].values)
z_all = np.apply_along_axis(unit_constrain, 0, z_unconstr_all)
