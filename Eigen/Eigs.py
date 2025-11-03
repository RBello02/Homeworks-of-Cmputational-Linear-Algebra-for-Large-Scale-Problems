import numpy as np
import pandas as pd
from IPython.display import display
# this class is used ti create a "superclass" (is not possible in reality) over np.linealg.eig to add a print method

class Eig:
    def __init__(self,A):
        self.A = A
        self.eigenvalues,self.eigenvectors = np.linalg.eig(self.A)

        idx = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]



 
    def _fmt(self, z):
        """Restituisce una stringa pulita con 2 cifre significative"""
        if abs(z.imag) < 1e-10:
            return f"{z.real:.2g}"
        else:
            return f"{z.real:.2g}+{z.imag:.2g}j"

    def to_dataframe(self):
        """Restituisce un DataFrame Pandas ben formattato"""
        # Prepara le intestazioni con autovalori e nomi colonna allineati
        headers = [f"v{i+1}\n(λ={self._fmt(val)})" for i, val in enumerate(self.eigenvalues)]
        
        fmt_vectors = np.vectorize(self._fmt)(self.eigenvectors)
        df = pd.DataFrame(fmt_vectors, columns=headers,
                          index=[f"x{i+1}" for i in range(self.eigenvectors.shape[0])])
        return df

    def show(self):
        """Mostra il risultato nel notebook"""
        print("Eigenvalues (sorted descending):")
        for i, val in enumerate(self.eigenvalues):
            print(f"  λ{i+1} = {self._fmt(val)}")
        print("\nEigenvectors (columns):")
        display(self.to_dataframe().style.set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]}]
        ).set_properties(**{'text-align': 'center'}))
