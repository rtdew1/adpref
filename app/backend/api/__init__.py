from config import PARADIGM

from api.shared.endpoints import *


if PARADIGM == "ar":
    from api.ar.endpoints import *
elif PARADIGM == "mpc":
    from api.mpc.endpoints import *
elif PARADIGM == "fe":
    from api.fe.endpoints import *
elif PARADIGM == "mpc_binary":
    from api.mpc_binary.endpoints import *

else:
    raise ModuleNotFoundError(f"Paradigm {PARADIGM} is not implemented")
