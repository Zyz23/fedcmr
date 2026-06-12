# -*- coding: utf-8 -*-
from pcode.masters.master_heterofl import MasterHeteroFL


class MasterFjORD(MasterHeteroFL):
    """FjORD server: distribute each client's maximal OD submodel.

    FjORD keeps a full server model, sends every selected client the largest
    ordered-dropout submodel it can train, and aggregates the returned nested
    parameters by overlap, as in HeteroFL-style ordered model families.
    """

    pass