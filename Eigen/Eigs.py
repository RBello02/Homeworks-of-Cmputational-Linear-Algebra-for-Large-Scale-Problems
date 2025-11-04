import numpy as np
import pandas as pd
from IPython.display import display
# this class is used ti create a "superclass" (is not possible in reality) over np.linealg.eig to add a print method

class Eig:
    def __init__(self,A):
        self.A = A
        self.values,self.vectors = np.linalg.eig(self.A)

        idx = np.argsort(self.values)[::-1]
        self.values = self.values[idx]
        self.vectors = self.vectors[:, idx]



 
    def _fmt(self, z):
        if abs(z.imag) < 1e-10:
            return f"{z.real:.2g}"
        else:
            return f"{z.real:.2g}+{z.imag:.2g}j"

    def to_dataframe(self):
        headers = [f"v{i+1}\n(λ={self._fmt(val)})" for i, val in enumerate(self.values)]
        
        fmt_vectors = np.vectorize(self._fmt)(self.vectors)
        df = pd.DataFrame(fmt_vectors, columns=headers,
                          index=[f"x{i+1}" for i in range(self.vectors.shape[0])])
        return df

    def show(self):
        print("values (sorted descending):")
        for i, val in enumerate(self.values):
            print(f"  λ{i+1} = {self._fmt(val)}")
        print("\nvectors (columns):")
        display(self.to_dataframe().style.set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]}]
        ).set_properties(**{'text-align': 'center'}))
