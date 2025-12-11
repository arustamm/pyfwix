from pysep3d.sep3d import Sep3DStep
import Hypercube
import SepVector
import numpy as np

class ConvertToSepVector (Sep3DStep):
    """Convert numpy arrays in 'data' column to SepVector objects"""
    def __init__(self, ns, ds, os, column='data'):
        self.ns = ns
        self.ds = ds
        self.os = os
        self.column = column
        super().__init__("ConvertToSepVector", ns=ns, ds=ds, os=os, column=column)
    
    def apply(self, ddf):
        def to_sepvector(df):
            axis = Hypercube.axis(n=self.ns, d=self.ds, o=self.os)
            hyper = Hypercube.hypercube(axes=[axis])
             # Convert each array to SepVector
            def array_to_vec(arr):
                vec = SepVector.getSepVector(hyper, storage='dataComplex')
                vec[:] = arr
                return vec
            
            df = df.copy()
            df[self.column] = df[self.column].apply(array_to_vec)

            return df
        
        return ddf.map_partitions(to_sepvector, meta=ddf._meta)