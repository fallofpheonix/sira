from pathlib import Path


class SymbolicExpression:
    def __init__(self, terms, variable_names):
        self.terms = terms  # list of (name, coef) tuples
        self.variable_names = variable_names

    def to_string(self):
        return " + ".join(f"{c:.4f}*{n}" for n, c in self.terms)

    def to_latex(self):
        parts = []
        for name, coef in self.terms:
            parts.append(f"{coef:.4f} \\cdot {name}")
        return " + ".join(parts)

    def to_callable(self):
        terms = self.terms
        def fn(**kwargs):
            result = 0.0
            for name, coef in terms:
                if name in kwargs:
                    result += coef * kwargs[name]
                elif name == '1':
                    result += coef
            return result
        return fn

    def rank_by_complexity(self):
        return sorted(self.terms, key=lambda x: abs(x[1]), reverse=True)


def save_expressions(expressions, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for name, expr in expressions.items():
            f.write(f"{name}: {expr}\n")
