

class HPCANDISC:
    """
    High Performance Canonical Discriminant Analysis (HPCANDISC)
    
    The HPCANDISC procedure is a high-performance procedure that performs canonical discriminant analysis. 
    It is a high-performance version of the CANDISC procedure.

    Parameters
    ----------
    

    
    """
    def __init__(
            self, n_components = 2, classes = None, n_workers=-1, chunk_size=100, max_chunks_per_worker=8,warn_message = True
    ):
        self.n_components = n_components
        self.classes = classes





        self.warn_message = warn_message

    def decision_function(self,X):
        pass
        
    def fit(self,X,y):
        pass

    def fit_transform(self,X,y):
        pass

    def transform(self,X):
        pass
