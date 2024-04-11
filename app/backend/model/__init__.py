from config import MODEL

if MODEL == "knn":
    from model.knn import PrefOptim

elif MODEL == "og":
    from model.og import PrefOptim

elif MODEL == "ucb":
    from model.ucb import PrefOptim

elif MODEL == "remboish":
    from model.remboish import PrefOptim

elif MODEL == "aggregated":
    from model.aggregated import PrefOptim

elif MODEL == "linear":
    from model.linear import PrefOptim

elif MODEL == "abernethy":
    from model.abernethy import PrefOptim

elif MODEL == "gur":
    from model.gur import PrefOptim

elif MODEL == "gur_sym":
    from model.gur_sym import PrefOptim

else:
    raise ModuleNotFoundError(f"Model '{MODEL}' is not implemented")
