from lark import Lark, Transformer, v_args
import numpy as np
import lorentz_structures as ls
import pathlib
import os


@v_args(inline=True)
class UFOTree(Transformer):
    from operator import add, sub, mul, truediv as div, neg, pos
    decimal = int
    real = float

    functions = {
        'complex': (lambda x, y: x + 1j*y),
        'C': ls.ChargeConj,
        'Identity': ls.Identity,
        'Gamma': ls.Gamma,
        'Gamma5': ls.Gamma5,
        'ProjP': ls.ProjP,
        'ProjM': ls.ProjM,
        'Epsilon': ls.Epsilon,
        'Metric': ls.Metric,
        'P': ls.Momentum,
        'cmath.sqrt': np.sqrt,
        'cmath.cos': np.cos,
        'cmath.sin': np.sin,
    }

    def __init__(self):
        self.vars = {'cmath.pi': np.pi}

    def call(self, name, idxs):
        funcname = '.'.join(name.children)
        return self.functions[funcname](*(idxs.children))

    def imaginary(self, value):
        return complex(value)

    def assign(self, name, value):
        self.vars[name] = value
        return value

    def var(self, name):
        varname = '.'.join(name.children)
        try:
            return self.vars[varname]
        except KeyError:
            return varname

    def pow(self, base, power=1):
        if power == 1:
            return base
        return base**power


class UFOParser:
    def __init__(self, model=None):
        cwd = pathlib.Path(__file__).parent.absolute()

        if model is not None:
            for function in model.all_functions:
                self._load_function(function)

        self.parser = Lark.open(os.path.join(cwd, 'ufo.lark'),
                                parser='lalr',
                                transformer=UFOTree())

    def _load_function(self, function):
        arc_trig_funcs = [
            ('acos', 'arccos'),
            ('asin', 'arcsin'),
            ('atan', 'arctan'),
        ]
        name = function.name
        arguments = ','.join(function.arguments)
        expression = function.expr.replace('cmath', 'np')
        for atrig in arc_trig_funcs:
            expression = expression.replace(atrig[0], atrig[1])
        UFOTree.functions[name] = eval(f'lambda {arguments}: {expression}')

    def __call__(self, expression):
        return self.parser.parse(expression)
