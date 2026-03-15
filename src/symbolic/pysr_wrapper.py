class PySRWrapper:
    def __init__(self):
        try:
            import pysr
            self.available = True
            self._pysr = pysr
        except ImportError:
            self.available = False
            self._pysr = None
        self._model = None

    def fit(self, X, y):
        if not self.available:
            raise ImportError("PySR not installed. Install with: pip install pysr")
        self._model = self._pysr.PySRRegressor()
        self._model.fit(X, y)
        return self

    def get_best_equation(self):
        if self._model is None:
            raise RuntimeError("Call fit() first")
        return str(self._model.sympy())
