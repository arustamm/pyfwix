from pysep3d.sep3d import ProcessingStep
import Hypercube
import SepVector
import numpy as np

class ConvertToSepVector(ProcessingStep):
    """
    Convert numpy arrays in 'data' column to SepVector objects.
    Dynamically determines 'ns' from array length and 'os' from 'trace_origin'.
    """
    def __init__(self, ds, col='data', origin_col='trace_origin'):
        # We only pass 'ds' (delta) because 'ns' and 'os' vary per trace
        super().__init__("ConvertToSepVector", ds=ds, col=col, origin_col=origin_col)
        self.ds = ds
        self.col = col
        self.origin_col = origin_col
    
    def apply(self, ddf):
        
        def to_sepvector(df):
            if df.empty:
                return df
            
            # We must use a row-wise apply because 'os' changes for every row
            def row_to_vec(row):
                # 1. Get dynamic parameters
                arr = row[self.col]
                ns = len(arr)
                os = row[self.origin_col]
                
                # 2. Construct Hypercube for this specific trace
                axis = Hypercube.axis(n=ns, d=self.ds, o=os)
                hyper = Hypercube.hypercube(axes=[axis])
                
                # 3. Create SepVector
                vec = SepVector.getSepVector(hyper, storage='dataComplex')
                
                # 4. Fill data (using getNdArray ensures we hit the buffer)
                # Ensure input is complex, matching storage='dataComplex'
                vec.getNdArray()[:] = arr.astype(np.complex64)
                return vec
            
            df = df.copy()
            # Apply the function to every row (axis=1)
            df[self.col] = df.apply(row_to_vec, axis=1)

            return df
        
        # ddf._meta needs to reflect that 'data' is now an object (the SepVector)
        # It likely already is 'object' if it contained numpy arrays, but good to be safe.
        return ddf.map_partitions(to_sepvector, meta=ddf._meta)