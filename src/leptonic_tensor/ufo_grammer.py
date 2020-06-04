from lark import Lark, Transformer, v_args
import numpy as np
from . import lorentz_structures as ls

ufo_grammer = """
    ?start: expression

    // Expressions
    ?expression: arith
    ?arith: term | term "+" arith                   -> add
        | term "-" arith                            -> sub
    ?term: factor | factor "*" term                 -> mul
        | term "/" term                             -> div
    ?factor: atom | "+" factor                      -> pos
        | "-" factor                                -> neg
    ?atom: "(" expression ")"
        | function
        | NUMBER "j"                                -> imaginary
        | NUMBER                                    -> number

    // function evaluation
    ?function: funcname "(" indices ")"             -> func_eval
    ?funcname: NAME

    // Helper functions
    ?indices: (NUMBER ("," NUMBER)*)                -> indices

    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.WS_INLINE

    %ignore WS_INLINE
"""


@v_args(inline=True)
class UFOTree(Transformer):
    from operator import add, sub, mul, truediv as div, neg, pos
    number = float
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
    }

    def __init__(self):
        self.vars = {}

    def indices(self, *args):
        return [int(i) for i in args]

    def func_eval(self, name, idxs):
        return self.functions[name](*idxs)

    def imaginary(self, value):
        return float(value)*1j


ufo_parser = Lark(ufo_grammer, parser='lalr', transformer=UFOTree())
ufo = ufo_parser.parse
