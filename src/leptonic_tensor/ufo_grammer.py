from lark import Lark, Transformer, v_args
import numpy as np
from . import lorentz_structures as ls

ufo_grammer = """
    ?start: expression

    // Expressions
    ?expression: arith
        | func_def ":=" func_expr                   -> register_function
        | NAME ":=" func_expr                       -> register

    ?arith: term | term "+" arith                   -> add
        | term "-" arith                            -> sub
    ?term: factor | factor "*" term                 -> mul
        | term "/" term                             -> div
    ?factor: atom | "+" factor                      -> pos
        | "-" factor                                -> neg
    ?atom: "(" arith ")"
        | function
        | NUMBER "j"                                -> imaginary
        | NUMBER                                    -> number
        | NAME                                      -> var

    // functions
    ?func_def: funcname "(" arglist ")"
    ?function: funcname "(" indices ")"             -> func_eval
    ?funcname: NAME
    ?func_expr: term_expr
        | term_expr "+" func_expr                   -> add
        | term_expr "-" func_expr                   -> sub
    ?term_expr: factor_expr
        | factor_expr "*" term_expr                 -> mul
        | term_expr "/" term_expr                   -> div
    ?factor_expr: atom_expr
        | "+" factor_expr                           -> pos
        | "-" factor_expr                           -> neg
    ?atom_expr: "(" func_expr ")"
        | function
        | NUMBER "j"                                -> imaginary
        | NUMBER                                    -> number
        | NAME


    // Helper functions
    ?arglist: (NAME ("," NAME)*)                    -> args
    ?indices: (wnumber ("," wnumber)*)              -> indices
    ?wnumber: NUMBER | "-" NUMBER

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

    def args(self, *args):
        return [str(i) for i in args]

    def func_eval(self, name, idxs):
        return self.functions[name](*idxs)

    def imaginary(self, value):
        return float(value)*1j

    def register(self, name, value):
        self.vars[name] = value
        return value

    def register_function(self, name, func):
        print(name.children[0], name.children[1], func)
        func_name = name.children[0]
        func_args = name.children[1]
        self.functions[name.children[0]] = lambda *func_args: func
        print(self.functions[func_name](1, 1))

    def var(self, name):
        return self.vars[name]


ufo_parser = Lark(ufo_grammer, parser='lalr', transformer=UFOTree())
ufo = ufo_parser.parse
