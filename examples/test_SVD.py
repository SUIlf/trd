import numpy as np
from TRD.decompositions import TRSVD



data = np.arange(1.,361.).reshape((3,4,5,6)).copy()
tr_core, tr_rank = TRSVD(data)