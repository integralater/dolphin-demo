from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import re

# ================== Expression 기본 클래스 ==================

class Expression(ABC):
    """수식을 표현하는 추상 기본 클래스"""

    @abstractmethod
    def __str__(self) -> str:
        """Expression을 문자열로 변환"""
        pass

    @abstractmethod
    def to_tree_node(self) -> dict:
        """Tree 시각화를 위한 노드 데이터 반환"""
        pass

    def __repr__(self) -> str:
        """개발자를 위한 표현(Expression 구조 표시)"""
        return self._repr_helper()

    @abstractmethod
    def _repr_helper(self) -> str:
        """repr 구현을 위한 helper 메소드"""
        pass

# ================== Atom 클래스들 ==================

class None_(Expression):
    """None을 표현하는 클래스 (None은 Python 예약어이므로 None_로 명명)"""

    def __str__(self) -> str:
        return ""

    def to_tree_node(self) -> dict:
        return {"label": "None", "children": []}

    def _repr_helper(self) -> str:
        return "None_()"

class Value(Expression):
    """숫자나 변수 값을 표현하는 클래스"""

    def __init__(self, val):
        self.val = val

    def __str__(self) -> str:
        return str(self.val)

    def to_tree_node(self) -> dict:
        return {"label": f"Value({self.val})", "children": []}

    def _repr_helper(self) -> str:
        return f"Value({self.val})"

class Text(Expression):
    """문자열을 표현하는 클래스"""

    def __init__(self, text: str):
        self.text = text

    def __str__(self) -> str:
        return self.text

    def to_tree_node(self) -> dict:
        return {"label": f'Text("{self.text}")', "children": []}

    def _repr_helper(self) -> str:
        return f'Text("{self.text}")'

class RecurringDecimal(Expression):
    """순환소수를 표현하는 클래스"""

    def __init__(self, non_recurring: Expression, recurring: Expression):
        self.non_recurring = non_recurring
        self.recurring = recurring

    def __str__(self) -> str:
        s = str(self.non_recurring)
        dot = "" if "." in s else "."
        return f"{s}{dot}\\bar{{{self.recurring}}}"

    def to_tree_node(self) -> dict:
        return {"label": "RecurringDecimal", "children": [
            self.non_recurring.to_tree_node(),
            self.recurring.to_tree_node()
        ]}

    def _repr_helper(self) -> str:
        return f"RecurringDecimal({self.non_recurring._repr_helper()}, {self.recurring._repr_helper()})"

# ================== 0항 연산자 (상수/기호) ==================

class EmptySet(Expression):
    """공집합을 표현하는 클래스"""

    def __str__(self) -> str:
        return "∅"

    def to_tree_node(self) -> dict:
        return {"label": "EmptySet", "children": []}

    def _repr_helper(self) -> str:
        return "EmptySet()"

class Infty(Expression):
    """무한대를 표현하는 클래스"""

    def __str__(self) -> str:
        return "∞"

    def to_tree_node(self) -> dict:
        return {"label": "Infty", "children": []}

    def _repr_helper(self) -> str:
        return "Infty()"

class UpArrow(Expression):
    """위쪽 화살표를 표현하는 클래스"""

    def __str__(self) -> str:
        return "↑"

    def to_tree_node(self) -> dict:
        return {"label": "UpArrow", "children": []}

    def _repr_helper(self) -> str:
        return "UpArrow()"

class DownArrow(Expression):
    """아래쪽 화살표를 표현하는 클래스"""

    def __str__(self) -> str:
        return "↓"

    def to_tree_node(self) -> dict:
        return {"label": "DownArrow", "children": []}

    def _repr_helper(self) -> str:
        return "DownArrow()"

class LeftArrow(Expression):
    """왼쪽 화살표를 표현하는 클래스"""

    def __str__(self) -> str:
        return "←"

    def to_tree_node(self) -> dict:
        return {"label": "LeftArrow", "children": []}

    def _repr_helper(self) -> str:
        return "LeftArrow()"

class RightArrow(Expression):
    """오른쪽 화살표를 표현하는 클래스"""

    def __str__(self) -> str:
        return "→"

    def to_tree_node(self) -> dict:
        return {"label": "RightArrow", "children": []}

    def _repr_helper(self) -> str:
        return "RightArrow()"

class Cdots(Expression):
    """줄임표(...)를 표현하는 클래스"""

    def __str__(self) -> str:
        return "…"

    def to_tree_node(self) -> dict:
        return {"label": "Cdots", "children": []}

    def _repr_helper(self) -> str:
        return "Cdots()"

class Square(Expression):
    """사각형 기호를 표현하는 클래스"""

    def __str__(self) -> str:
        return "□"

    def to_tree_node(self) -> dict:
        return {"label": "Square", "children": []}

    def _repr_helper(self) -> str:
        return "Square()"

class Circ(Expression):
    """원 기호를 표현하는 클래스"""

    def __str__(self) -> str:
        return "○"

    def to_tree_node(self) -> dict:
        return {"label": "Circ", "children": []}

    def _repr_helper(self) -> str:
        return "Circ()"

class BigCircle(Expression):
    """큰 원 기호를 표현하는 클래스"""

    def __str__(self) -> str:
        return "∘"

    def to_tree_node(self) -> dict:
        return {"label": "BigCircle", "children": []}

    def _repr_helper(self) -> str:
        return "BigCircle()"

class Degree(Expression):
    """도(degree) 기호를 표현하는 클래스"""

    def __str__(self) -> str:
        return "°"

    def to_tree_node(self) -> dict:
        return {"label": "Degree", "children": []}

    def _repr_helper(self) -> str:
        return "Degree()"

class EulerNum(Expression):
    """오일러 수 e를 표현하는 클래스"""

    def __str__(self) -> str:
        return "e"

    def to_tree_node(self) -> dict:
        return {"label": "EulerNum", "children": []}

    def _repr_helper(self) -> str:
        return "EulerNum()"

# ================== 단항 연산자 ==================

class Absolute(Expression):
    """절댓값을 표현하는 클래스"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def __str__(self) -> str:
        return f"|{self.expr}|"

    def to_tree_node(self) -> dict:
        return {"label": "Absolute", "children": [self.expr.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Absolute({self.expr._repr_helper()})"

class Factorial(Expression):
    """팩토리얼을 표현하는 클래스"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def __str__(self) -> str:
        return f"({self.expr})!"

    def to_tree_node(self) -> dict:
        return {"label": "Factorial", "children": [self.expr.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Factorial({self.expr._repr_helper()})"

class Plus(Expression):
    """단항 플러스를 표현하는 클래스"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def __str__(self) -> str:
        return f"+{self.expr}"

    def to_tree_node(self) -> dict:
        return {"label": "Plus", "children": [self.expr.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Plus({self.expr._repr_helper()})"

class Minus(Expression):
    """단항 마이너스를 표현하는 클래스"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def __str__(self) -> str:
        return f"-{self.expr}"

    def to_tree_node(self) -> dict:
        return {"label": "Minus", "children": [self.expr.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Minus({self.expr._repr_helper()})"

class Not(Expression):
    """논리 부정을 표현하는 클래스"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def __str__(self) -> str:
        return f"~{self.expr}"

    def to_tree_node(self) -> dict:
        return {"label": "Not", "children": [self.expr.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Not({self.expr._repr_helper()})"

class SetComple(Expression):
    """집합의 여집합을 표현하는 클래스"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def __str__(self) -> str:
        return f"{self.expr}ᶜ"

    def to_tree_node(self) -> dict:
        return {"label": "SetComple", "children": [self.expr.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"SetComple({self.expr._repr_helper()})"

class SetNum(Expression):
    """집합의 원소 개수를 표현하는 클래스"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def __str__(self) -> str:
        return f"n({self.expr})"

    def to_tree_node(self) -> dict:
        return {"label": "SetNum", "children": [self.expr.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"SetNum({self.expr._repr_helper()})"

class Delta(Expression):
    """변화량을 표현하는 클래스"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def __str__(self) -> str:
        return f"Δ{self.expr}"

    def to_tree_node(self) -> dict:
        return {"label": "Delta", "children": [self.expr.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Delta({self.expr._repr_helper()})"

class Bar(Expression):
    """평균이나 켤레복소수를 표현하는 클래스"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def __str__(self) -> str:
        return f"̄\\bar{{{self.expr}}}"

    def to_tree_node(self) -> dict:
        return {"label": "Bar", "children": [self.expr.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Bar({self.expr._repr_helper()})"

class Vec(Expression):
    """벡터를 표현하는 클래스"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def __str__(self) -> str:
        return f"\\vec{{{self.expr}}}"

    def to_tree_node(self) -> dict:
        return {"label": "Vec", "children": [self.expr.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Vec({self.expr._repr_helper()})"

class Norm(Expression):
    """벡터의 크기를 표현하는 클래스"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def __str__(self) -> str:
        return f"|{self.expr}|"

    def to_tree_node(self) -> dict:
        return {"label": "Norm", "children": [self.expr.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Norm({self.expr._repr_helper()})"

class Prime(Expression):
    """미분 연산자(프라임)를 표현하는 클래스"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def __str__(self) -> str:
        # 항상 (expr)' 형태로 표현
        return f"({self.expr})'"

    def to_tree_node(self) -> dict:
        return {"label": "Prime", "children": [self.expr.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Prime({self.expr._repr_helper()})"

class Gauss(Expression):
    """가우스 함수(바닥 함수)를 표현하는 클래스"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def __str__(self) -> str:
        return f"[{self.expr}]"

    def to_tree_node(self) -> dict:
        return {"label": "Gauss", "children": [self.expr.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Gauss({self.expr._repr_helper()})"

class Unit(Expression):
    """단위를 표현하는 클래스"""

    def __init__(self, unit: str):
        self.unit = unit

    def __str__(self) -> str:
        return self.unit

    def to_tree_node(self) -> dict:
        return {"label": f"Unit({self.unit})", "children": []}

    def _repr_helper(self) -> str:
        return f"Unit({self.unit})"

class Triangle(Expression):
    """삼각형을 표현하는 클래스"""

    def __init__(self, vertices: str):
        self.vertices = vertices

    def __str__(self) -> str:
        return f"△{self.vertices}"

    def to_tree_node(self) -> dict:
        return {"label": f"Triangle({self.vertices})", "children": []}

    def _repr_helper(self) -> str:
        return f"Triangle({self.vertices})"

class Angle(Expression):
    """각을 표현하는 클래스"""

    def __init__(self, vertices: str):
        self.vertices = vertices

    def __str__(self) -> str:
        return f"∠{self.vertices}"

    def to_tree_node(self) -> dict:
        return {"label": f"Angle({self.vertices})", "children": []}

    def _repr_helper(self) -> str:
        return f"Angle({self.vertices})"

class Arc(Expression):
    """호를 표현하는 클래스"""

    def __init__(self, vertices: str):
        self.vertices = vertices

    def __str__(self) -> str:
        return f"\\overset{{\\frown}}{{{self.vertices}}}"

    def to_tree_node(self) -> dict:
        return {"label": f"Arc({self.vertices})", "children": []}

    def _repr_helper(self) -> str:
        return f"Arc({self.vertices})"

# ================== 2항 연산자 ==================

class Add(Expression):
    """덧셈을 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left}+{self.right})"

    def to_tree_node(self) -> dict:
        return {"label": "Add", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Add({self.left._repr_helper()}, {self.right._repr_helper()})"

class Sub(Expression):
    """뺄셈을 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left}-{self.right})"

    def to_tree_node(self) -> dict:
        return {"label": "Sub", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Sub({self.left._repr_helper()}, {self.right._repr_helper()})"

class Mult(Expression):
    """곱셈을 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left}×{self.right})"

    def to_tree_node(self) -> dict:
        return {"label": "Mult", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Mult({self.left._repr_helper()}, {self.right._repr_helper()})"

class ImplicitMult(Expression):
    """생략된 곱셈을 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left}{self.right})"

    def to_tree_node(self) -> dict:
        return {"label": "ImplicitMult", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"ImplicitMult({self.left._repr_helper()}, {self.right._repr_helper()})"

class Divide(Expression):
    """나눗셈(÷)을 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left}÷{self.right})"

    def to_tree_node(self) -> dict:
        return {"label": "Divide", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Divide({self.left._repr_helper()}, {self.right._repr_helper()})"

class PlusMinus(Expression):
    """플러스마이너스(±)를 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left}±{self.right})"

    def to_tree_node(self) -> dict:
        return {"label": "PlusMinus", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"PlusMinus({self.left._repr_helper()}, {self.right._repr_helper()})"

class MinusPlus(Expression):
    """마이너스플러스(∓)를 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left}∓{self.right})"

    def to_tree_node(self) -> dict:
        return {"label": "MinusPlus", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"MinusPlus({self.left._repr_helper()}, {self.right._repr_helper()})"

class Slash(Expression):
    """슬래시(/)를 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left}/{self.right})"

    def to_tree_node(self) -> dict:
        return {"label": "Slash", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Slash({self.left._repr_helper()}, {self.right._repr_helper()})"

class Frac(Expression):
    """분수(LaTeX \\frac{}{})를 표현하는 클래스"""

    def __init__(self, num: Expression, denom: Expression):
        self.num = num
        self.denom = denom

    def __str__(self) -> str:
        return f"\\frac{{{self.num}}}{{{self.denom}}}"

    def to_tree_node(self) -> dict:
        return {"label": "Frac", "children": [self.denom.to_tree_node(), self.num.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Frac({self.num._repr_helper()}, {self.denom._repr_helper()})"

class Power(Expression):
    """거듭제곱을 표현하는 클래스"""

    def __init__(self, base: Expression, expo: Expression):
        self.base = base
        self.expo = expo

    def __str__(self) -> str:
        expo_str = str(self.expo)
        
        # 삼각함수의 거듭제곱 처리
        if isinstance(self.base, (Sin, Cos, Tan, Sec, Csc, Cot)):
            trig_name = type(self.base).__name__.lower()  # 'sin', 'cos', etc.
            arg = self.base.arg
            
            if expo_str == "2":
                return f"{trig_name}²({arg})"
            elif expo_str == "3":
                return f"{trig_name}³({arg})"
            else:
                return f"{trig_name}^{{{self.expo}}}({arg})"
        
        # 일반적인 거듭제곱 처리
        if expo_str == "2":
            return f"({self.base}²)"
        elif expo_str == "3":
            return f"({self.base}³)"
        else:
            return f"{{{self.base}}}^{{{self.expo}}}"

    def to_tree_node(self) -> dict:
        return {"label": "Power", "children": [self.base.to_tree_node(), self.expo.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Power({self.base._repr_helper()}, {self.expo._repr_helper()})"

class Less(Expression):
    """부등호(<)를 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{self.left}<{self.right}"

    def to_tree_node(self) -> dict:
        return {"label": "Less", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Less({self.left._repr_helper()}, {self.right._repr_helper()})"

class Leq(Expression):
    """부등호(≤)를 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{self.left}≤{self.right}"

    def to_tree_node(self) -> dict:
        return {"label": "Leq", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Leq({self.left._repr_helper()}, {self.right._repr_helper()})"

class Greater(Expression):
    """부등호(>)를 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{self.left}>{self.right}"

    def to_tree_node(self) -> dict:
        return {"label": "Greater", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Greater({self.left._repr_helper()}, {self.right._repr_helper()})"

class Geq(Expression):
    """부등호(≥)를 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{self.left}≥{self.right}"

    def to_tree_node(self) -> dict:
        return {"label": "Geq", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Geq({self.left._repr_helper()}, {self.right._repr_helper()})"

class Eq(Expression):
    """등호(=)를 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{self.left}={self.right}"

    def to_tree_node(self) -> dict:
        return {"label": "Eq", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Eq({self.left._repr_helper()}, {self.right._repr_helper()})"

class Neq(Expression):
    """부등호(≠)를 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{self.left}≠{self.right}"

    def to_tree_node(self) -> dict:
        return {"label": "Neq", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Neq({self.left._repr_helper()}, {self.right._repr_helper()})"

class Subscript(Expression):
    """아래첨자를 표현하는 클래스"""

    def __init__(self, base: Expression, sub: Expression):
        self.base = base
        self.sub = sub

    def __str__(self) -> str:
        return f"{self.base}_{{{self.sub}}}"

    def to_tree_node(self) -> dict:
        return {"label": "Subscript", "children": [self.base.to_tree_node(), self.sub.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Subscript({self.base._repr_helper()}, {self.sub._repr_helper()})"

class Superscript(Expression):
    """위첨자를 표현하는 클래스"""

    def __init__(self, base: Expression, sup: Expression):
        self.base = base
        self.sup = sup

    def __str__(self) -> str:
        return f"{self.base}^{{{self.sup}}}"

    def to_tree_node(self) -> dict:
        return {"label": "Superscript", "children": [self.base.to_tree_node(), self.sup.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Superscript({self.base._repr_helper()}, {self.sup._repr_helper()})"

class SetIn(Expression):
    """원소 포함을 표현하는 클래스"""

    def __init__(self, elem: Expression, set_expr: Expression):
        self.elem = elem
        self.set_expr = set_expr

    def __str__(self) -> str:
        return f"{self.elem}∈{self.set_expr}"

    def to_tree_node(self) -> dict:
        return {"label": "SetIn", "children": [self.elem.to_tree_node(), self.set_expr.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"SetIn({self.elem._repr_helper()}, {self.set_expr._repr_helper()})"

class SetNotIn(Expression):
    """원소 미포함을 표현하는 클래스"""

    def __init__(self, elem: Expression, set_expr: Expression):
        self.elem = elem
        self.set_expr = set_expr

    def __str__(self) -> str:
        return f"{self.elem}∉{self.set_expr}"

    def to_tree_node(self) -> dict:
        return {"label": "SetNotIn", "children": [self.elem.to_tree_node(), self.set_expr.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"SetNotIn({self.elem._repr_helper()}, {self.set_expr._repr_helper()})"

class SetSub(Expression):
    """부분집합을 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{self.left}⊂{self.right}"

    def to_tree_node(self) -> dict:
        return {"label": "SetSub", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"SetSub({self.left._repr_helper()}, {self.right._repr_helper()})"

class SetNotSub(Expression):
    """부분집합이 아님을 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{self.left}⊄{self.right}"

    def to_tree_node(self) -> dict:
        return {"label": "SetNotSub", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"SetNotSub({self.left._repr_helper()}, {self.right._repr_helper()})"

class SetSup(Expression):
    """부분집합을 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{self.left}⊃{self.right}"

    def to_tree_node(self) -> dict:
        return {"label": "SetSup", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"SetSup({self.left._repr_helper()}, {self.right._repr_helper()})"

class SetNotSup(Expression):
    """부분집합이 아님을 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{self.left}⊅{self.right}"

    def to_tree_node(self) -> dict:
        return {"label": "SetNotSup", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"SetNotSup({self.left._repr_helper()}, {self.right._repr_helper()})"

class SetCup(Expression):
    """합집합을 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left}∪{self.right})"

    def to_tree_node(self) -> dict:
        return {"label": "SetCup", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"SetCup({self.left._repr_helper()}, {self.right._repr_helper()})"

class SetCap(Expression):
    """교집합을 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left}∩{self.right})"

    def to_tree_node(self) -> dict:
        return {"label": "SetCap", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"SetCap({self.left._repr_helper()}, {self.right._repr_helper()})"

class Rimpl(Expression):
    """오른쪽 함의를 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left}→{self.right})"

    def to_tree_node(self) -> dict:
        return {"label": "Rimpl", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Rimpl({self.left._repr_helper()}, {self.right._repr_helper()})"

class Limpl(Expression):
    """왼쪽 함의를 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left}←{self.right})"

    def to_tree_node(self) -> dict:
        return {"label": "Limpl", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Limpl({self.left._repr_helper()}, {self.right._repr_helper()})"

class Biconditional(Expression):
    """필요충분조건(↔)을 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left}↔{self.right})"

    def to_tree_node(self) -> dict:
        return {"label": "Biconditional", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Biconditional({self.left._repr_helper()}, {self.right._repr_helper()})"

class Iff(Expression):
    """참/거짓을 강조한 필요충분조건(⇔)"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left}⇔{self.right})"

    def to_tree_node(self) -> dict:
        return {"label": "Iff", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Iff({self.left._repr_helper()}, {self.right._repr_helper()})"

class Rsufficient(Expression):
    """오른쪽 충분조건을 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left}⇒{self.right})"

    def to_tree_node(self) -> dict:
        return {"label": "Rsufficient", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Rsufficient({self.left._repr_helper()}, {self.right._repr_helper()})"

class Lsufficient(Expression):
    """왼쪽 충분조건을 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left}⇐{self.right})"

    def to_tree_node(self) -> dict:
        return {"label": "Lsufficient", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Lsufficient({self.left._repr_helper()}, {self.right._repr_helper()})"

class Prop(Expression):
    """명제를 표현하는 클래스"""

    def __init__(self, symbol: Expression, statement: Expression):
        self.symbol = symbol
        self.statement = statement

    def __str__(self) -> str:
        return f"{self.symbol}:{self.statement}"

    def to_tree_node(self) -> dict:
        return {"label": "Prop", "children": [self.symbol.to_tree_node(), self.statement.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Prop({self.symbol._repr_helper()}, {self.statement._repr_helper()})"

class FuncDef(Expression):
    """함수 정의를 표현하는 클래스"""

    def __init__(self, func: Expression, domain: Expression, codomain: Expression):
        self.func = func
        self.domain = domain
        self.codomain = codomain

    def __str__(self) -> str:
        return f"{self.func}:{self.domain}→{self.codomain}"

    def to_tree_node(self) -> dict:
        return {"label": "FuncDef", "children": [self.func.to_tree_node(), self.domain.to_tree_node(), self.codomain.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"FuncDef({self.func._repr_helper()}, {self.domain._repr_helper()}, {self.codomain._repr_helper()})"

class Perm(Expression):
    """순열을 표현하는 클래스"""

    def __init__(self, n: Expression, r: Expression):
        self.n = n
        self.r = r

    def __str__(self) -> str:
        return f"_{self.n}P_{self.r}"

    def to_tree_node(self) -> dict:
        return {"label": "Perm", "children": [self.n.to_tree_node(), self.r.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Perm({self.n._repr_helper()}, {self.r._repr_helper()})"

class Comb(Expression):
    """조합을 표현하는 클래스"""

    def __init__(self, n: Expression, r: Expression):
        self.n = n
        self.r = r

    def __str__(self) -> str:
        return f"_{self.n}C_{self.r}"

    def to_tree_node(self) -> dict:
        return {"label": "Comb", "children": [self.n.to_tree_node(), self.r.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Comb({self.n._repr_helper()}, {self.r._repr_helper()})"

class RepeatedPermu(Expression):
    """중복순열을 표현하는 클래스"""

    def __init__(self, n: Expression, r: Expression):
        self.n = n
        self.r = r

    def __str__(self) -> str:
        return f"_{self.n}Π_{self.r}"

    def to_tree_node(self) -> dict:
        return {"label": "RepeatedPermu", "children": [self.n.to_tree_node(), self.r.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"RepeatedPermu({self.n._repr_helper()}, {self.r._repr_helper()})"

class RepeatedComb(Expression):
    """중복조합을 표현하는 클래스"""

    def __init__(self, n: Expression, r: Expression):
        self.n = n
        self.r = r

    def __str__(self) -> str:
        return f"_{self.n}H_{self.r}"

    def to_tree_node(self) -> dict:
        return {"label": "RepeatedComb", "children": [self.n.to_tree_node(), self.r.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"RepeatedComb({self.n._repr_helper()}, {self.r._repr_helper()})"

class Log(Expression):
    """로그를 표현하는 클래스"""

    def __init__(self, base: Expression, arg: Expression):
        self.base = base
        self.arg = arg

    def __str__(self) -> str:
        return f"log_{self.base}({self.arg})"

    def to_tree_node(self) -> dict:
        return {"label": "Log", "children": [self.base.to_tree_node(), self.arg.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Log({self.base._repr_helper()}, {self.arg._repr_helper()})"

class SQRT(Expression):
    """제곱근을 표현하는 클래스"""

    def __init__(self, radicand: Expression, index: Expression = None):
        self.radicand = radicand
        self.index = index if index is not None else Value(2)

    def __str__(self) -> str:
        if str(self.index) == "2":
            return f"√{{{self.radicand}}}"
        return f"\\sqrt[{self.index}]{{{self.radicand}}}"

    def to_tree_node(self) -> dict:
        return {"label": "SQRT", "children": [self.radicand.to_tree_node(), self.index.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"SQRT({self.radicand._repr_helper()}, {self.index._repr_helper()})"

class SetBuilder(Expression):
    """조건제시법 집합을 표현하는 클래스"""

    def __init__(self, var: Expression, condition: Expression):
        self.var = var
        self.condition = condition

    def __str__(self) -> str:
        return f"{{{self.var}|{self.condition}}}"

    def to_tree_node(self) -> dict:
        return {"label": "SetBuilder", "children": [self.var.to_tree_node(), self.condition.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"SetBuilder({self.var._repr_helper()}, {self.condition._repr_helper()})"

class LineExpr(Expression):
    """직선 방정식을 표현하는 클래스"""

    def __init__(self, line: Expression, eq: Expression):
        self.line = line
        self.eq = eq

    def __str__(self) -> str:
        return f"{self.line}:{self.eq}"

    def to_tree_node(self) -> dict:
        return {"label": "LineExpr", "children": [self.line.to_tree_node(), self.eq.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"LineExpr({self.line._repr_helper()}, {self.eq._repr_helper()})"

class Ratio(Expression):
    """비율을 표현하는 클래스 (1:2, 1:2:3 등 다중 비율 지원)"""

    def __init__(self, ratios: List[Expression]):
        self.ratios = ratios

    def __str__(self) -> str:
        return ":".join(str(r) for r in self.ratios)

    def to_tree_node(self) -> dict:
        return {"label": "Ratio", "children": [r.to_tree_node() for r in self.ratios]}

    def _repr_helper(self) -> str:
        ratio_strs = [r._repr_helper() for r in self.ratios]
        return f"Ratio([{', '.join(ratio_strs)}])"

class Segment(Expression):
    """선분을 표현하는 클래스"""

    def __init__(self, start: Expression, end: Expression):
        self.start = start
        self.end = end

    def __str__(self) -> str:
        return f"̄\\bar{{{self.start}{self.end}}}"

    def to_tree_node(self) -> dict:
        return {"label": "Segment", "children": [self.start.to_tree_node(), self.end.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Segment({self.start._repr_helper()}, {self.end._repr_helper()})"

class Ray(Expression):
    """반직선을 표현하는 클래스"""

    def __init__(self, start: Expression, through: Expression):
        self.start = start
        self.through = through

    def __str__(self) -> str:
        return f"\\overrightarrow{{{self.start}{self.through}}}"

    def to_tree_node(self) -> dict:
        return {"label": "Ray", "children": [self.start.to_tree_node(), self.through.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Ray({self.start._repr_helper()}, {self.through._repr_helper()})"

class Line(Expression):
    """직선을 표현하는 클래스"""

    def __init__(self, point1: Expression, point2: Expression):
        self.point1 = point1
        self.point2 = point2

    def __str__(self) -> str:
        return f"\\overleftrightarrow{{{self.point1}{self.point2}}}"

    def to_tree_node(self) -> dict:
        return {"label": "Line", "children": [self.point1.to_tree_node(), self.point2.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Line({self.point1._repr_helper()}, {self.point2._repr_helper()})"

class Perp(Expression):
    """수직을 표현하는 클래스"""

    def __init__(self, line1: Expression, line2: Expression):
        self.line1 = line1
        self.line2 = line2

    def __str__(self) -> str:
        return f"{self.line1}⊥{self.line2}"

    def to_tree_node(self) -> dict:
        return {"label": "Perp", "children": [self.line1.to_tree_node(), self.line2.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Perp({self.line1._repr_helper()}, {self.line2._repr_helper()})"

class Paral(Expression):
    """평행을 표현하는 클래스"""

    def __init__(self, line1: Expression, line2: Expression):
        self.line1 = line1
        self.line2 = line2

    def __str__(self) -> str:
        return f"{self.line1}∥{self.line2}"

    def to_tree_node(self) -> dict:
        return {"label": "Paral", "children": [self.line1.to_tree_node(), self.line2.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Paral({self.line1._repr_helper()}, {self.line2._repr_helper()})"

class InnerProduct(Expression):
    """내적을 표현하는 클래스"""

    def __init__(self, vec1: Expression, vec2: Expression):
        self.vec1 = vec1
        self.vec2 = vec2

    def __str__(self) -> str:
        return f"({self.vec1}·{self.vec2})"

    def to_tree_node(self) -> dict:
        return {"label": "InnerProduct", "children": [self.vec1.to_tree_node(), self.vec2.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"InnerProduct({self.vec1._repr_helper()}, {self.vec2._repr_helper()})"

class Congru(Expression):
    """합동을 표현하는 클래스"""

    def __init__(self, shape1: Expression, shape2: Expression):
        self.shape1 = shape1
        self.shape2 = shape2

    def __str__(self) -> str:
        return f"{self.shape1}≡{self.shape2}"

    def to_tree_node(self) -> dict:
        return {"label": "Congru", "children": [self.shape1.to_tree_node(), self.shape2.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Congru({self.shape1._repr_helper()}, {self.shape2._repr_helper()})"

class Sim(Expression):
    """닮음을 표현하는 클래스"""

    def __init__(self, shape1: Expression, shape2: Expression):
        self.shape1 = shape1
        self.shape2 = shape2

    def __str__(self) -> str:
        return f"{self.shape1}∼{self.shape2}"

    def to_tree_node(self) -> dict:
        return {"label": "Sim", "children": [self.shape1.to_tree_node(), self.shape2.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Sim({self.shape1._repr_helper()}, {self.shape2._repr_helper()})"

class UnitDiv(Expression):
    """단위 나눗셈을 표현하는 클래스"""

    def __init__(self, num_unit: Expression, denom_unit: Expression):
        self.num_unit = num_unit
        self.denom_unit = denom_unit

    def __str__(self) -> str:
        return f"{self.num_unit}/{self.denom_unit}"

    def to_tree_node(self) -> dict:
        return {"label": "UnitDiv", "children": [self.num_unit.to_tree_node(), self.denom_unit.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"UnitDiv({self.num_unit._repr_helper()}, {self.denom_unit._repr_helper()})"

class UnitMult(Expression):
    """단위 곱셈을 표현하는 클래스"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{self.left} {self.right}"

    def to_tree_node(self) -> dict:
        return {"label": "UnitMult", "children": [self.left.to_tree_node(), self.right.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"UnitMult({self.left._repr_helper()}, {self.right._repr_helper()})"

class About(Expression):
    """근사값을 표현하는 클래스"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def __str__(self) -> str:
        return f"약 {self.expr}"

    def to_tree_node(self) -> dict:
        return {"label": "About", "children": [self.expr.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"About({self.expr._repr_helper()})"

class Point(Expression):
    """좌표 평면의 점을 표현하는 클래스 (n차원 지원)"""

    def __init__(self, name: Optional[Expression], args: List[Expression]):
        self.name = name
        self.args = args  # List[Expression]로 n차원 좌표 처리

    def __str__(self) -> str:
        # 좌표 문자열 생성
        coords_str = ", ".join(str(arg) for arg in self.args)
        
        # 이름이 있는 경우
        if self.name is not None and not isinstance(self.name, None_):
            return f"점 {self.name}({coords_str})"
        
        # 이름이 없는 경우
        return f"({coords_str})"

    def to_tree_node(self) -> dict:
        children = []
        
        # name이 None이 아닌 경우만 추가
        if self.name is not None and not isinstance(self.name, None_):
            children.append(self.name.to_tree_node())
        
        # 모든 좌표 추가
        children.extend([arg.to_tree_node() for arg in self.args])
        
        return {"label": "Point", "children": children}

    def _repr_helper(self) -> str:
        args_repr = ", ".join(arg._repr_helper() for arg in self.args)
        
        if self.name is not None and not isinstance(self.name, None_):
            return f"Point({self.name._repr_helper()}, [{args_repr}])"
        
        return f"Point(None, [{args_repr}])"

# ================== 3항 연산자 ==================

class MixedFrac(Expression):
    """대분수를 표현하는 클래스"""

    def __init__(self, whole: Expression, num: Expression, denom: Expression):
        self.whole = whole
        self.num = num
        self.denom = denom

    def __str__(self) -> str:
        return f"{self.whole}\\frac{{{self.num}}}{{{self.denom}}}"

    def to_tree_node(self) -> dict:
        return {
            "label": "MixedFrac",
            "children": [
                self.whole.to_tree_node(),
                self.num.to_tree_node(),
                self.denom.to_tree_node()
            ]
        }

    def _repr_helper(self) -> str:
        return f"MixedFrac({self.whole._repr_helper()}, {self.num._repr_helper()}, {self.denom._repr_helper()})"

class Diff(Expression):
    """미분을 표현하는 클래스"""

    def __init__(self, y: Expression, x: Expression, n: Expression = None):
        self.y = y
        self.x = x
        self.n = n if n is not None else Value(1)

    def __str__(self) -> str:
        if str(self.n) == "1":
            return f"d{self.y}/d{self.x}"
        return f"d^{self.n}{self.y}/d{self.x}^{self.n}"

    def to_tree_node(self) -> dict:
        return {
            "label": "Diff",
            "children": [
                self.y.to_tree_node(),
                self.x.to_tree_node(),
                self.n.to_tree_node()
            ]
        }

    def _repr_helper(self) -> str:
        return f"Diff({self.y._repr_helper()}, {self.x._repr_helper()}, {self.n._repr_helper()})"

# ================== 가변 인자 연산자 ==================

class Func(Expression):
    """함수를 표현하는 클래스"""

    def __init__(self, name: Expression, args: List[Expression]):
        self.name = name
        self.args = args

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.name}({args_str})"

    def to_tree_node(self) -> dict:
        children = [self.name.to_tree_node()] + [arg.to_tree_node() for arg in self.args]
        return {"label": "Func", "children": children}

    def _repr_helper(self) -> str:
        args_repr = ", ".join(arg._repr_helper() for arg in self.args)
        return f"Func({self.name._repr_helper()}, [{args_repr}])"

class FuncInv(Expression):
    """역함수를 표현하는 클래스"""

    def __init__(self, name: Expression, args: List[Expression]):
        self.name = name
        self.args = args

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.name}^(-1)({args_str})"

    def to_tree_node(self) -> dict:
        children = [self.name.to_tree_node()] + [arg.to_tree_node() for arg in self.args]
        return {"label": "FuncInv", "children": children}

    def _repr_helper(self) -> str:
        args_repr = ", ".join(arg._repr_helper() for arg in self.args)
        return f"FuncInv({self.name._repr_helper()}, [{args_repr}])"

class SetRoster(Expression):
    """원소나열법 집합을 표현하는 클래스"""

    def __init__(self, elements: List[Expression]):
        self.elements = elements

    def __str__(self) -> str:
        elements_str = ", ".join(str(elem) for elem in self.elements)
        return f"{{{elements_str}}}"

    def to_tree_node(self) -> dict:
        children = [elem.to_tree_node() for elem in self.elements]
        return {"label": "SetRoster", "children": children}

    def _repr_helper(self) -> str:
        elements_repr = ", ".join(elem._repr_helper() for elem in self.elements)
        return f"SetRoster([{elements_repr}])"

class Cases(Expression):
    """연립방정식을 표현하는 클래스"""

    def __init__(self, cases: List[Tuple[Expression, Expression]]):
        self.cases = cases

    def __str__(self) -> str:
        cases_str = "\n".join(f"{expr}, ({cond})" for expr, cond in self.cases)
        return f"{{\n{cases_str}\n}}"

    def to_tree_node(self) -> dict:
        children = []
        for expr, cond in self.cases:
            children.append(expr.to_tree_node())
            children.append(cond.to_tree_node())
        return {"label": "Cases", "children": children}

    def _repr_helper(self) -> str:
        cases_repr = ", ".join(f"({expr._repr_helper()}, {cond._repr_helper()})"
                              for expr, cond in self.cases)
        return f"Cases([{cases_repr}])"

class Seq(Expression):
    """수열을 표현하는 클래스"""

    def __init__(self, term: Expression):
        self.term = term

    def __str__(self) -> str:
        return f"{{{self.term}}}"

    def to_tree_node(self) -> dict:
        return {"label": "Seq", "children": [self.term.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Seq({self.term._repr_helper()})"

class Lim(Expression):
    """극한을 표현하는 클래스. 좌극한, 우극한은 Add(x, None_()), Sub(x, None_())으로 처리하면 됨."""

    def __init__(self, var: Expression, to: Expression, expr: Expression):
        self.var = var
        self.to = to
        self.expr = expr

    def __str__(self) -> str:
        return f"lim_{{{self.var}→{self.to}}} {self.expr}"

    def to_tree_node(self) -> dict:
        return {
            "label": "Lim",
            "children": [
                self.var.to_tree_node(),
                self.to.to_tree_node(),
                self.expr.to_tree_node()
            ]
        }

    def _repr_helper(self) -> str:
        return f"Lim({self.var._repr_helper()}, {self.to._repr_helper()}, {self.expr._repr_helper()})"

class Sum(Expression):
    """급수를 표현하는 클래스"""

    def __init__(self, term: Expression, var: Expression, start: Expression, end: Expression):
        self.term = term
        self.var = var
        self.start = start
        self.end = end

    def __str__(self) -> str:
        return f"Σ_{{{self.var}={self.start}}}^{{{self.end}}} ({self.term})"

    def to_tree_node(self) -> dict:
        return {
            "label": "Sum",
            "children": [
                self.term.to_tree_node(),
                self.var.to_tree_node(),
                self.start.to_tree_node(),
                self.end.to_tree_node()
            ]
        }

    def _repr_helper(self) -> str:
        return f"Sum({self.term._repr_helper()}, {self.var._repr_helper()}, {self.start._repr_helper()}, {self.end._repr_helper()})"

class Integral(Expression):
    """적분을 표현하는 클래스"""

    def __init__(self, lower: Expression, upper: Expression, integrand: Expression, var: Expression):
        self.lower = lower
        self.upper = upper
        self.integrand = integrand
        self.var = var

    def __str__(self) -> str:
        if isinstance(self.lower, None_) and isinstance(self.upper, None_):
            return f"∫{self.integrand}d{self.var}"
        return f"∫_{{{self.lower}}}^{{{self.upper}}}{self.integrand}d{self.var}"

    def to_tree_node(self) -> dict:
        return {
            "label": "Integral",
            "children": [
                self.lower.to_tree_node(),
                self.upper.to_tree_node(),
                self.integrand.to_tree_node(),
                self.var.to_tree_node()
            ]
        }

    def _repr_helper(self) -> str:
        return f"Integral({self.lower._repr_helper()}, {self.upper._repr_helper()}, {self.integrand._repr_helper()}, {self.var._repr_helper()})"

class Integrated(Expression):
    """정적분 계산 결과를 표현하는 클래스"""

    def __init__(self, antiderivative: Expression, lower: Expression, upper: Expression):
        self.antiderivative = antiderivative
        self.lower = lower
        self.upper = upper

    def __str__(self) -> str:
        return f"[{self.antiderivative}]_{{{self.lower}}}^{{{self.upper}}}"

    def to_tree_node(self) -> dict:
        return {
            "label": "Integrated",
            "children": [
                self.antiderivative.to_tree_node(),
                self.lower.to_tree_node(),
                self.upper.to_tree_node()
            ]
        }

    def _repr_helper(self) -> str:
        return f"Integrated({self.antiderivative._repr_helper()}, {self.lower._repr_helper()}, {self.upper._repr_helper()})"

class Prob(Expression):
    """확률을 표현하는 클래스"""

    def __init__(self, event: Expression, condition: Expression = None):
        self.event = event
        self.condition = condition

    def __str__(self) -> str:
        # condition이 Python None 이거나, None_ 인스턴스인 경우 => P(event)
        if self.condition is None or isinstance(self.condition, None_):
            return f"P({self.event})"
        return f"P({self.event}|{self.condition})"

    def to_tree_node(self) -> dict:
        children = [self.event.to_tree_node()]
        if self.condition is not None and not isinstance(self.condition, None_):
            children.append(self.condition.to_tree_node())
        return {"label": "Prob", "children": children}

    def _repr_helper(self) -> str:
        if self.condition is None or isinstance(self.condition, None_):
            return f"Prob({self.event._repr_helper()})"
        return f"Prob({self.event._repr_helper()}, {self.condition._repr_helper()})"

# ================== 구간 ==================

class ClosedInterval(Expression):
    """닫힌 구간 [A, B]를 표현하는 클래스 (A 이상 B 이하)"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"[{self.left}, {self.right}]"

    def to_tree_node(self) -> dict:
        return {"label": "ClosedInterval", "children": [
            self.left.to_tree_node(),
            self.right.to_tree_node()
        ]}

    def _repr_helper(self) -> str:
        return f"ClosedInterval({self.left._repr_helper()}, {self.right._repr_helper()})"

class ClosedOpenInterval(Expression):
    """닫힌-열린 구간 [A, B)를 표현하는 클래스 (A 이상 B 미만)"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"[{self.left}, {self.right})"

    def to_tree_node(self) -> dict:
        return {"label": "ClosedOpenInterval", "children": [
            self.left.to_tree_node(),
            self.right.to_tree_node()
        ]}

    def _repr_helper(self) -> str:
        return f"ClosedOpenInterval({self.left._repr_helper()}, {self.right._repr_helper()})"

class OpenClosedInterval(Expression):
    """열린-닫힌 구간 (A, B]를 표현하는 클래스 (A 초과 B 이하)"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left}, {self.right}]"

    def to_tree_node(self) -> dict:
        return {"label": "OpenClosedInterval", "children": [
            self.left.to_tree_node(),
            self.B.to_tree_node()
        ]}

    def _repr_helper(self) -> str:
        return f"OpenClosedInterval({self.left._repr_helper()}, {self.right._repr_helper()})"

class OpenInterval(Expression):
    """열린 구간 (A, B)를 표현하는 클래스 (A 초과 B 미만)"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left}, {self.right})"

    def to_tree_node(self) -> dict:
        return {"label": "OpenInterval", "children": [
            self.left.to_tree_node(),
            self.right.to_tree_node()
        ]}

    def _repr_helper(self) -> str:
        return f"OpenInterval({self.left._repr_helper()}, {self.right._repr_helper()})"

# ================== 삼각함수 ==================

class Sin(Expression):
    """사인 함수"""

    def __init__(self, arg: Expression):
        self.arg = arg

    def __str__(self) -> str:
        return f"sin({self.arg})"

    def to_tree_node(self) -> dict:
        return {"label": "Sin", "children": [self.arg.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Sin({self.arg._repr_helper()})"

class Cos(Expression):
    """코사인 함수"""

    def __init__(self, arg: Expression):
        self.arg = arg

    def __str__(self) -> str:
        return f"cos({self.arg})"

    def to_tree_node(self) -> dict:
        return {"label": "Cos", "children": [self.arg.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Cos({self.arg._repr_helper()})"

class Tan(Expression):
    """탄젠트 함수"""

    def __init__(self, arg: Expression):
        self.arg = arg

    def __str__(self) -> str:
        return f"tan({self.arg})"

    def to_tree_node(self) -> dict:
        return {"label": "Tan", "children": [self.arg.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Tan({self.arg._repr_helper()})"

class Sec(Expression):
    """시컨트 함수"""

    def __init__(self, arg: Expression):
        self.arg = arg

    def __str__(self) -> str:
        return f"sec({self.arg})"

    def to_tree_node(self) -> dict:
        return {"label": "Sec", "children": [self.arg.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Sec({self.arg._repr_helper()})"

class Csc(Expression):
    """코시컨트 함수"""

    def __init__(self, arg: Expression):
        self.arg = arg

    def __str__(self) -> str:
        return f"csc({self.arg})"

    def to_tree_node(self) -> dict:
        return {"label": "Csc", "children": [self.arg.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Csc({self.arg._repr_helper()})"

class Cot(Expression):
    """코탄젠트 함수"""

    def __init__(self, arg: Expression):
        self.arg = arg

    def __str__(self) -> str:
        return f"cot({self.arg})"

    def to_tree_node(self) -> dict:
        return {"label": "Cot", "children": [self.arg.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Cot({self.arg._repr_helper()})"

class Ln(Expression):
    """자연로그 함수"""

    def __init__(self, arg: Expression):
        self.arg = arg

    def __str__(self) -> str:
        return f"ln({self.arg})"

    def to_tree_node(self) -> dict:
        return {"label": "Ln", "children": [self.arg.to_tree_node()]}

    def _repr_helper(self) -> str:
        return f"Ln({self.arg._repr_helper()})"

class InlinePhrase(Expression):
    def __init__(self, parts: List[Expression]):
        self.parts = parts  # Text(), Value(), 기타 Expression 혼합 가능

    def __str__(self) -> str:
        # 기본은 그대로 이어 붙임(공백 정책은 Text 쪽에 위임)
        return "".join(str(p) for p in self.parts)

    def to_tree_node(self) -> dict:
        return {"label": "InlinePhrase", "children": [p.to_tree_node() for p in self.parts]}

    def _repr_helper(self) -> str:
        parts_repr = ", ".join(p._repr_helper() for p in self.parts)
        return f"InlinePhrase([{parts_repr}])"


# ================== Tree 시각화 함수 ==================

def visualize_expression_tree(expr: Expression, figsize=(12, 8), node_radius=0.3, vertical_spacing=3.0):
    """
    Expression Tree를 matplotlib으로 시각화
    - node_radius: 노드 원의 반지름
    - vertical_spacing: 노드 반지름에 곱해지는 세로 간격 계수
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')

    def calculate_tree_size(node):
        """트리의 크기와 깊이 계산"""
        if not node['children']:
            return 1, 1
        width, max_depth = 0, 0
        for child in node['children']:
            child_width, child_depth = calculate_tree_size(child)
            width += child_width
            max_depth = max(max_depth, child_depth)
        return width, max_depth + 1

    def draw_node_recursive(node, x, y, width):
        """재귀적으로 노드와 연결선 그리기"""
        # 노드 그리기
        circle = plt.Circle((x, y), node_radius, color='lightblue', ec='black', linewidth=2)
        ax.add_patch(circle)

        # 라벨 (node_radius에 따라 fontsize 조정)
        label = node['label']
        if len(label) > 12:
            label = label[:11] + '...'
        ax.text(x, y, label, ha='center', va='center',
                fontsize=max(6, int(node_radius * 20)), weight='bold')

        if node['children']:
            child_count = len(node['children'])

            # 자식 노드의 X 좌표 분포
            if child_count == 1:
                child_positions = [x]
            elif child_count == 2:
                offset = width / 4
                child_positions = [x - offset, x + offset]
            else:
                offset = width / (child_count + 1)
                child_positions = [x + (i - (child_count-1)/2) * offset for i in range(child_count)]

            for i, child in enumerate(node['children']):
                child_x = child_positions[i]
                # 부모와 자식 간 세로 거리 = 노드 반지름에 비례
                child_y = y - node_radius * vertical_spacing

                # 연결선 (노드 반지름 고려해서 끝점 조정)
                ax.plot([x, child_x],
                        [y - node_radius, child_y + node_radius],
                        'k-', linewidth=1)

                # 재귀적으로 자식 노드 그리기
                child_width, _ = calculate_tree_size(child)
                draw_node_recursive(child, child_x, child_y, width / child_count)

    # 트리 데이터 가져오기
    tree_data = expr.to_tree_node()
    tree_width, tree_depth = calculate_tree_size(tree_data)

    # 캔버스 크기 (세로축도 node_radius 기반으로 확장)
    ax.set_xlim(-tree_width - 1, tree_width + 1)
    ax.set_ylim(-tree_depth * node_radius * vertical_spacing - node_radius, node_radius * 2)

    # 트리 그리기 시작
    draw_node_recursive(tree_data, 0, 0, tree_width * 2)

    plt.title("Expression Tree Visualization", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()

# ================== 한글 대체 텍스트 변환 함수 ==================

def number_to_korean(input_value):
    """입
    력값을 한국어 숫자로 변환하는 함수"""
    small_units = ["일", "십", "백", "천"]
    big_units = ["", "만", "억", "조", "경", "해", "자", "양", "구", "간",
                 "정", "재", "극", "항하사", "아승기", "나유타", "불가사의", "무량대수"]
    digit_to_korean = {
        '0': '영', '1': '일', '2': '이', '3': '삼', '4': '사',
        '5': '오', '6': '육', '7': '칠', '8': '팔', '9': '구'
    }

    if isinstance(input_value, float):
        temp_str = repr(input_value)
        if 'e' in temp_str.lower():
            temp_str = format(input_value, '.15f')
            if '.' in temp_str:
                temp_str = temp_str.rstrip('0').rstrip('.')
        number_str = temp_str
    else:
        number_str = str(input_value)
        if 'e' in number_str.lower():
            try:
                float_val = float(number_str)
                temp_str = format(float_val, '.15f')
                if '.' in temp_str:
                    temp_str = temp_str.rstrip('0').rstrip('.')
                number_str = temp_str
            except:
                pass

    number_str = number_str.replace(",", "")
    is_negative = number_str.startswith('-')
    if is_negative:
        number_str = number_str[1:]
    elif number_str.startswith('+'):
        number_str = number_str[1:]

    if '.' in number_str:
        integer_part_temp, decimal_part_temp = number_str.split('.')
        integer_part_temp = integer_part_temp.lstrip('0') or '0'
        number_str = integer_part_temp + '.' + decimal_part_temp
    else:
        number_str = number_str.lstrip('0') or '0'

    result = "마이너스 " if is_negative else ""

    if '.' in number_str:
        integer_part, decimal_part = number_str.split('.')
    else:
        integer_part = number_str
        decimal_part = None

    if integer_part == '0' or integer_part == '':
        result += "영"
    else:
        groups = []
        temp = integer_part
        while temp:
            if len(temp) > 4:
                groups.append(temp[-4:])
                temp = temp[:-4]
            else:
                groups.append(temp)
                break
        groups.reverse()
        
        if len(groups) > len(big_units):
            return "숫자가 너무 커서 한국어로 읽을 수 없습니다."

        korean_parts = []
        for i, group in enumerate(groups):
            if group == '0000':
                continue
            group_korean = convert_four_digit_group(group, digit_to_korean, small_units)
            if group_korean:
                big_unit_index = len(groups) - 1 - i
                if big_unit_index < len(big_units):
                    big_unit = big_units[big_unit_index]
                    korean_parts.append(group_korean + big_unit)
                else:
                    korean_parts.append(group_korean)
        result += " ".join(korean_parts)

    if decimal_part is not None:
        result += " 점 "
        for digit in decimal_part:
            if digit in digit_to_korean:
                result += digit_to_korean[digit]

    return result.strip()


def convert_four_digit_group(group, digit_to_korean, small_units):
    """4자리 이하의 자연수를 한국어로 변환"""
    group = group.lstrip('0')
    if not group or group == '0':
        return ""
    result_parts = []
    group_len = len(group)
    for i, digit in enumerate(group):
        if digit == '0':
            continue
        digit_korean = digit_to_korean[digit]
        position = group_len - 1 - i
        if position == 0:
            result_parts.append(digit_korean)
        else:
            unit = small_units[position]
            if digit == '1':
                result_parts.append(unit)
            else:
                result_parts.append(digit_korean + unit)
    return "".join(result_parts)


def get_korean_alphabet(char):
    """알파벳을 한글로 변환"""
    alphabet_map = {
        'a': '에이', 'b': '비', 'c': '씨', 'd': '디', 'e': '이', 'f': '에프',
        'g': '지', 'h': '에이치', 'i': '아이', 'j': '제이', 'k': '케이',
        'l': '엘', 'm': '엠', 'n': '엔', 'o': '오', 'p': '피', 'q': '큐',
        'r': '알', 's': '에스', 't': '티', 'u': '유', 'v': '브이',
        'w': '더블유', 'x': '엑스', 'y': '와이', 'z': '제트'
    }
    greek_map = {
        'α': '알파', 'β': '베타', 'γ': '감마', 'δ': '델타', 'ε': '입실론',
        'ζ': '제타', 'η': '에타', 'θ': '세타', 'ι': '아이오타', 'κ': '카파',
        'λ': '람다', 'μ': '뮤', 'ν': '뉴', 'ξ': '크사이', 'ο': '오미크론',
        'π': '파이', 'ρ': '로', 'σ': '시그마', 'τ': '타우', 'υ': '입실론',
        'φ': '파이', 'χ': '카이', 'ψ': '프사이', 'ω': '오메가'
    }
    if char in greek_map:
        return greek_map[char]
    elif char.lower() in alphabet_map:
        if char.isupper():
            return "대문자 " + alphabet_map[char.lower()]
        return alphabet_map[char.lower()]
    return char


def get_particle(word):
    if not word:
        return "은", "이", "을", "과", "으로"
    last_char = word[-1]

    # 한글 받침 규칙
    if '가' <= last_char <= '힣':
        code = ord(last_char) - ord('가')
        jong = code % 28
        # jong == 0: 받침 없음 → 는/가/를/와/로
        if jong == 0:
            return "는", "가", "를", "와", "로"
        # ㄹ 받침(인덱스 8) → '로' 예외
        if jong == 8:
            return "은", "이", "을", "과", "로"
        # 그 밖의 받침 → '으로'
        return "은", "이", "을", "과", "으로"

    # 영문자: 모음 끝이면 로, 자음 끝이면 으로
    elif last_char.isalpha():
        vowels = 'aeiouAEIOU'
        if last_char in vowels:
            return "는", "가", "를", "와", "로"
        else:
            return "은", "이", "을", "과", "으로"

    return "은", "이", "을", "과", "으로"



def is_descriptive_operator(expr):
    """서술형 연산자인지 확인 (~이다, ~이고로 끝나는 연산자들)"""
    return isinstance(expr, (Less, Leq, Greater, Geq, Eq, Neq, 
                            SetIn, SetNotIn, SetSub, SetNotSub, SetSup, SetNotSup,
                            Paral, Perp, Congru, Sim))


def get_leftmost_operand(expr):
    """서술형 연산자가 중첩된 경우 가장 왼쪽 피연산자를 재귀적으로 추출"""
    if not is_descriptive_operator(expr):
        return expr
    
    # 각 연산자 타입에 따라 left 속성 이름이 다름
    if isinstance(expr, (Paral, Perp)):
        return get_leftmost_operand(expr.line1)
    elif isinstance(expr, (Congru, Sim)):
        return get_leftmost_operand(expr.shape1)
    elif isinstance(expr, (SetIn, SetNotIn)):
        return get_leftmost_operand(expr.elem)
    else:  # Less, Leq, Greater, Geq, Eq, Neq, SetSub, SetNotSub, SetSup, SetNotSup
        return get_leftmost_operand(expr.left)


def expression_to_korean(expr, is_nested=False, is_naive=False):
    """
    Expression을 한글 대체 텍스트로 변환
    
    Args:
        expr: 변환할 Expression 객체
        is_nested: 중첩된 표현인지 여부
        is_naive: True일 경우 실무자 스타일로 형식적 수식어 생략
                 (확률, 수열, 자연상수, 점 등의 접두어 제거)
    """
    
    # 0항 연산자
    if isinstance(expr, None_):
        return ""
    if isinstance(expr, Value):
        val_str = str(expr.val)
        if val_str.replace('.', '').replace('-', '').isdigit():
            return number_to_korean(expr.val)
        if len(val_str) == 1:
            return get_korean_alphabet(val_str)
        return " ".join(get_korean_alphabet(c) if c.isalpha() else c for c in val_str)
    if isinstance(expr, Text):
        return "".join(c for c in expr.text)
    if isinstance(expr, RecurringDecimal):
        non_rec_kor = expression_to_korean(expr.non_recurring, is_naive=is_naive)
        non_rec_raw = str(expr.non_recurring)
        need_dot = '.' not in non_rec_raw

        rec_expr = expr.recurring
        if isinstance(rec_expr, Value):
            rec_raw = str(rec_expr.val)
        else:
            rec_raw = str(rec_expr)

        if rec_raw.isdigit():
            # 숫자만으로 구성된 순환마디는 한 자리씩 읽음
            rec_kor = "".join(number_to_korean(ch).replace(" ", "") for ch in rec_raw)
        else:
            # 숫자 외가 섞였으면 일반 규칙으로 재귀 처리(탈락 방지)
            rec_kor = expression_to_korean(rec_expr, is_naive=is_naive)

        prefix = f"{non_rec_kor} 점 " if need_dot else f"{non_rec_kor} "
        return f"{prefix}순환마디 {rec_kor}"

    if isinstance(expr, EmptySet):
        return "공집합"
    if isinstance(expr, Infty):
        return "무한"
    if isinstance(expr, UpArrow):
        return "위화살표"
    if isinstance(expr, DownArrow):
        return "아래화살표"
    if isinstance(expr, LeftArrow):
        return "왼쪽화살표"
    if isinstance(expr, RightArrow):
        return "오른쪽화살표"
    if isinstance(expr, Cdots):
        return "쩜쩜쩜"
        
    # Unit 처리
    if isinstance(expr, Unit):
        # Unit_dictionary에서 발음 찾기
        try:
            import csv
            import os
            dict_path = os.path.join(os.path.dirname(__file__), 'Unit_dictionary.csv')
            with open(dict_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['표기법'] == expr.unit:
                        return row['한국어 발음']
        except Exception as e:
            pass  # 파일이 없거나 오류가 발생하면 기본값 사용
        return expr.unit  # 기본값
    if isinstance(expr, Square):
        return "네모"
    if isinstance(expr, Circ):
        return "도"
    if isinstance(expr, BigCircle):
        return "동그라미"
    if isinstance(expr, EulerNum):
        return "이" if is_naive else "자연상수 이"
    
    # 단항 연산자
    if isinstance(expr, Absolute):
        return f"절댓값 {expression_to_korean(expr.expr, is_naive=is_naive)}"
    if isinstance(expr, Factorial):
        return f"{expression_to_korean(expr.expr, is_naive=is_naive)} 팩토리얼"
    if isinstance(expr, Plus):
        return f"플러스 {expression_to_korean(expr.expr, is_naive=is_naive)}"
    if isinstance(expr, Minus):
        return f"마이너스 {expression_to_korean(expr.expr, is_naive=is_naive)}"
    if isinstance(expr, Not):
        return f"낫 {expression_to_korean(expr.expr, is_naive=is_naive)}"
    if isinstance(expr, SetComple):
        return f"{expression_to_korean(expr.expr, is_naive=is_naive)}의 여집합"
    if isinstance(expr, SetNum):
        return f"{expression_to_korean(expr.expr, is_naive=is_naive)}의 원소의 개수"
    if isinstance(expr, Delta):
        return f"델타 {expression_to_korean(expr.expr, is_naive=is_naive)}"
    if isinstance(expr, Bar):
        return f"{expression_to_korean(expr.expr, is_naive=is_naive)} 바"
    if isinstance(expr, Vec):
        return f"벡터 {expression_to_korean(expr.expr, is_naive=is_naive)}"
    if isinstance(expr, Norm):
        return f"노름 {expression_to_korean(expr.expr, is_naive=is_naive)}"
    if isinstance(expr, Triangle):
        return f"삼각형 {expr.vertices}"
    if isinstance(expr, Angle):
        return f"각 {expr.vertices}"
    if isinstance(expr, Arc):
        return f"호 {expr.vertices}"
    
    # 2항 연산자
    if isinstance(expr, Add):
        if isinstance(expr.right, None_):
            return f"{expression_to_korean(expr.left, is_naive=is_naive)} 플러스"
        return f"{expression_to_korean(expr.left, is_naive=is_naive)} 더하기 {expression_to_korean(expr.right, is_naive=is_naive)}"
    if isinstance(expr, Sub):
        if isinstance(expr.right, None_):
            return f"{expression_to_korean(expr.left, is_naive=is_naive)} 마이너스"
        return f"{expression_to_korean(expr.left, is_naive=is_naive)} 빼기 {expression_to_korean(expr.right, is_naive=is_naive)}"
    if isinstance(expr, Mult):
        return f"{expression_to_korean(expr.left, is_naive=is_naive)} 곱하기 {expression_to_korean(expr.right, is_naive=is_naive)}"
    if isinstance(expr, ImplicitMult):
        return f"{expression_to_korean(expr.left, is_naive=is_naive)} {expression_to_korean(expr.right, is_naive=is_naive)}"
    if isinstance(expr, Divide):
        return f"{expression_to_korean(expr.left, is_naive=is_naive)} 나누기 {expression_to_korean(expr.right, is_naive=is_naive)}"
    if isinstance(expr, PlusMinus):
        return f"{expression_to_korean(expr.left, is_naive=is_naive)} 플마 {expression_to_korean(expr.right, is_naive=is_naive)}"
    if isinstance(expr, MinusPlus):
        return f"{expression_to_korean(expr.left, is_naive=is_naive)} 마플 {expression_to_korean(expr.right, is_naive=is_naive)}"
    if isinstance(expr, Slash):
        return f"{expression_to_korean(expr.right, is_naive=is_naive)} 분의 {expression_to_korean(expr.left, is_naive=is_naive)}"
    if isinstance(expr, Frac):
        return f"{expression_to_korean(expr.denom, is_naive=is_naive)} 분의 {expression_to_korean(expr.num, is_naive=is_naive)}"
    if isinstance(expr, Gauss):
        return f"가우스 {expression_to_korean(expr.expr, is_naive=is_naive)}"     
    if isinstance(expr, Prime):
        return f"{expression_to_korean(expr.expr, is_naive=is_naive)} 프라임"  
    if isinstance(expr, Power):
        # 각도 표기 특별 처리: Power(Value(number), Degree()) -> InlinePhrase로 취급
        if isinstance(expr.base, Value) and isinstance(expr.expo, Degree):
            return f"{expression_to_korean(expr.base, is_naive=is_naive)} 도"
        # 온도 단위 특별 처리: Power(Value(number), Unit("℃"/"℉")) -> "섭씨/화씨 number 도씨"
        if isinstance(expr.base, Value) and isinstance(expr.expo, Unit) and expr.expo.unit in ['℃', '℉']:
            temp_name = "섭씨" if expr.expo.unit == "℃" else "화씨"
            return f"{temp_name} {expression_to_korean(expr.base, is_naive=is_naive)} 도씨"

        if isinstance(expr.base, (Sin, Cos, Tan, Sec, Csc, Cot)):
            trig_map = {Sin: "사인", Cos: "코사인", Tan: "탄젠트",
                       Sec: "시컨트", Csc: "코시컨트", Cot: "코탄젠트"}
            trig_name = trig_map[type(expr.base)]
            arg = expression_to_korean(expr.base.arg, is_naive=is_naive)
            if str(expr.expo) == "2":
                return f"{trig_name} 제곱 {arg}"
            return f"{trig_name} {expression_to_korean(expr.expo, is_naive=is_naive)} 제곱 {arg}"
        base = expression_to_korean(expr.base, is_naive=is_naive)
        if str(expr.expo) == "2":
            return f"{base} 제곱"
        return f"{base}의 {expression_to_korean(expr.expo, is_naive=is_naive)} 제곱"
    
    # 비교 연산자
    if isinstance(expr, Less):
        left = expression_to_korean(expr.left, is_naive=is_naive)
        particles = get_particle(left)
        if is_descriptive_operator(expr.right):
            right_leftmost = expression_to_korean(get_leftmost_operand(expr.right), is_naive=is_naive)
            right_rest = expression_to_korean(expr.right, is_nested=True, is_naive=is_naive)
            return f"{left}{particles[1]} {right_leftmost}보다 작고 {right_rest}"
        right = expression_to_korean(expr.right, is_naive=is_naive)
        if is_nested:
            # 중첩된 경우, 자신의 right가 서술형이 아니면 마지막이므로 "작다"로 끝남
            return f"{left}{particles[1]} {right}보다 작다"
        return f"{left}{particles[1]} {right}보다 작다"
    
    if isinstance(expr, Leq):
        left = expression_to_korean(expr.left, is_naive=is_naive)
        particles = get_particle(left)
        if is_descriptive_operator(expr.right):
            right_leftmost = expression_to_korean(get_leftmost_operand(expr.right), is_naive=is_naive)
            right_rest = expression_to_korean(expr.right, is_nested=True, is_naive=is_naive)
            return f"{left}{particles[1]} {right_leftmost} 이하이고 {right_rest}"
        right = expression_to_korean(expr.right, is_naive=is_naive)
        if is_nested:
            return f"{left}{particles[1]} {right} 이하이다"
        return f"{left}{particles[1]} {right} 이하이다"
    
    if isinstance(expr, Greater):
        left = expression_to_korean(expr.left, is_naive=is_naive)
        particles = get_particle(left)
        if is_descriptive_operator(expr.right):
            right_leftmost = expression_to_korean(get_leftmost_operand(expr.right), is_naive=is_naive)
            right_rest = expression_to_korean(expr.right, is_nested=True, is_naive=is_naive)
            return f"{left}{particles[1]} {right_leftmost}보다 크고 {right_rest}"
        right = expression_to_korean(expr.right, is_naive=is_naive)
        if is_nested:
            return f"{left}{particles[1]} {right}보다 크다"
        return f"{left}{particles[1]} {right}보다 크다"
    
    if isinstance(expr, Geq):
        left = expression_to_korean(expr.left, is_naive=is_naive)
        particles = get_particle(left)
        if is_descriptive_operator(expr.right):
            right_leftmost = expression_to_korean(get_leftmost_operand(expr.right), is_naive=is_naive)
            right_rest = expression_to_korean(expr.right, is_nested=True, is_naive=is_naive)
            return f"{left}{particles[1]} {right_leftmost} 이상이고 {right_rest}"
        right = expression_to_korean(expr.right, is_naive=is_naive)
        if is_nested:
            return f"{left}{particles[1]} {right} 이상이다"
        return f"{left}{particles[1]} {right} 이상이다"
    
    if isinstance(expr, Eq):
        left = expression_to_korean(expr.left, is_naive=is_naive)
        particles = get_particle(left)
        if is_descriptive_operator(expr.right):
            right_leftmost = expression_to_korean(get_leftmost_operand(expr.right), is_naive=is_naive)
            right_rest = expression_to_korean(expr.right, is_nested=True, is_naive=is_naive)
            return f"{left}{particles[0]} {right_leftmost}이고 {right_rest}"
        right = expression_to_korean(expr.right, is_naive=is_naive)
        if is_nested:
            return f"{left}{particles[0]} {right}"
        return f"{left}{particles[0]} {right}"
    
    if isinstance(expr, Neq):
        left = expression_to_korean(expr.left, is_naive=is_naive)
        particles = get_particle(left)
        right = expression_to_korean(expr.right, is_naive=is_naive)
        if is_nested:
            return f"{left}{particles[0]} {right}가 아니다"
        return f"{left}{particles[0]} {right}가 아니다"
    
    if isinstance(expr, Subscript):
        return f"{expression_to_korean(expr.base, is_naive=is_naive)} {expression_to_korean(expr.sub, is_naive=is_naive)}"
    if isinstance(expr, Superscript):
        return f"{expression_to_korean(expr.base, is_naive=is_naive)} {expression_to_korean(expr.sup, is_naive=is_naive)}"
    
    # 집합 관계 연산자
    if isinstance(expr, SetIn):
        elem = expression_to_korean(expr.elem, is_naive=is_naive)
        particles = get_particle(elem)
        if is_descriptive_operator(expr.set_expr):
            set_leftmost = expression_to_korean(get_leftmost_operand(expr.set_expr), is_naive=is_naive)
            set_rest = expression_to_korean(expr.set_expr, is_nested=True, is_naive=is_naive)
            return f"{elem}{particles[0]} {set_leftmost}의 원소이고 {set_rest}"
        set_expr = expression_to_korean(expr.set_expr, is_naive=is_naive)
        if is_nested:
            return f"{elem}{particles[0]} {set_expr}의 원소이다"
        return f"{elem}{particles[0]} {set_expr}의 원소이다"
    
    if isinstance(expr, SetNotIn):
        elem = expression_to_korean(expr.elem, is_naive=is_naive)
        particles = get_particle(elem)
        set_expr = expression_to_korean(expr.set_expr, is_naive=is_naive)
        if is_nested:
            return f"{elem}{particles[0]} {set_expr}의 원소가 아니다"
        return f"{elem}{particles[0]} {set_expr}의 원소가 아니다"
    
    if isinstance(expr, SetSub):
        left = expression_to_korean(expr.left, is_naive=is_naive)
        particles = get_particle(left)
        if is_descriptive_operator(expr.right):
            right_leftmost = expression_to_korean(get_leftmost_operand(expr.right), is_naive=is_naive)
            right_rest = expression_to_korean(expr.right, is_nested=True, is_naive=is_naive)
            return f"{left}{particles[0]} {right_leftmost}의 부분집합이고 {right_rest}"
        right = expression_to_korean(expr.right, is_naive=is_naive)
        if is_nested:
            return f"{left}{particles[0]} {right}의 부분집합이다"
        return f"{left}{particles[0]} {right}의 부분집합이다"
    
    if isinstance(expr, SetNotSub):
        left = expression_to_korean(expr.left, is_naive=is_naive)
        particles = get_particle(left)
        right = expression_to_korean(expr.right, is_naive=is_naive)
        if is_nested:
            return f"{left}{particles[0]} {right}의 부분집합이 아니다"
        return f"{left}{particles[0]} {right}의 부분집합이 아니다"
    
    if isinstance(expr, SetSup):
        left = expression_to_korean(expr.left, is_naive=is_naive)
        particles = get_particle(left)
        if is_descriptive_operator(expr.right):
            right_leftmost = expression_to_korean(get_leftmost_operand(expr.right), is_naive=is_naive)
            right_rest = expression_to_korean(expr.right, is_nested=True, is_naive=is_naive)
            return f"{left}{particles[0]} {right_leftmost}을 포함하고 {right_rest}"
        right = expression_to_korean(expr.right, is_naive=is_naive)
        if is_nested:
            return f"{left}{particles[0]} {right}을 포함한다"
        return f"{left}{particles[0]} {right}을 포함한다"
    
    if isinstance(expr, SetNotSup):
        left = expression_to_korean(expr.left, is_naive=is_naive)
        particles = get_particle(left)
        right = expression_to_korean(expr.right, is_naive=is_naive)
        if is_nested:
            return f"{left}{particles[0]} {right}을 포함하지 않는다"
        return f"{left}{particles[0]} {right}을 포함하지 않는다"
    
    if isinstance(expr, SetCup):
        return f"{expression_to_korean(expr.left, is_naive=is_naive)} 합집합 {expression_to_korean(expr.right, is_naive=is_naive)}"
    if isinstance(expr, SetCap):
        return f"{expression_to_korean(expr.left, is_naive=is_naive)} 교집합 {expression_to_korean(expr.right, is_naive=is_naive)}"
    if isinstance(expr, Rimpl):
        return f"{expression_to_korean(expr.left, is_naive=is_naive)}이면 {expression_to_korean(expr.right, is_naive=is_naive)}"
    if isinstance(expr, Limpl):
        left = expression_to_korean(expr.left, is_naive=is_naive)
        particles = get_particle(left)
        return f"{left}{particles[0]} {expression_to_korean(expr.right, is_naive=is_naive)}에 의해 함의된다"
    if isinstance(expr, Biconditional):
        left = expression_to_korean(expr.left, is_naive=is_naive)
        particles = get_particle(left)
        right = expression_to_korean(expr.right, is_naive=is_naive)
        return f"{left}{particles[0]} {right}의 필요충분조건이다"
    if isinstance(expr, Iff):
        left = expression_to_korean(expr.left, is_naive=is_naive)
        particles = get_particle(left)
        right = expression_to_korean(expr.right, is_naive=is_naive)
        return f"참으로 {left}{particles[0]} {right}의 필요충분조건이다"
    if isinstance(expr, Rsufficient):
        left = expression_to_korean(expr.left, is_naive=is_naive)
        particles = get_particle(left)
        return f"{left}{particles[0]} {expression_to_korean(expr.right, is_naive=is_naive)}의 충분조건이다"
    if isinstance(expr, Lsufficient):
        left = expression_to_korean(expr.left, is_naive=is_naive)
        particles = get_particle(left)
        return f"{left}{particles[0]} {expression_to_korean(expr.right, is_naive=is_naive)}의 필요조건이다"
    if isinstance(expr, Prop):
        return f"명제 {expression_to_korean(expr.symbol, is_naive=is_naive)} {expression_to_korean(expr.statement, is_naive=is_naive)}"
    if isinstance(expr, FuncDef):
        func = expression_to_korean(expr.func, is_naive=is_naive)
        domain = expression_to_korean(expr.domain, is_naive=is_naive)
        codomain = expression_to_korean(expr.codomain, is_naive=is_naive)
        nen, _, _, _, euro_codomain = get_particle(func)[0], *get_particle(func)[1:]
        # codomain 뒤에는 '로/으로' 필요
        euro = get_particle(codomain)[4]
        return f"함수 {func}{get_particle(func)[0]} {domain}에서 {codomain}{euro} 가는 함수"

    if isinstance(expr, Perm):
        return f"순열 {expression_to_korean(expr.n, is_naive=is_naive)} 피 {expression_to_korean(expr.r, is_naive=is_naive)}"
    if isinstance(expr, Comb):
        return f"조합 {expression_to_korean(expr.n, is_naive=is_naive)} 씨 {expression_to_korean(expr.r, is_naive=is_naive)}"
    if isinstance(expr, RepeatedPermu):
        return f"중복순열 {expression_to_korean(expr.n, is_naive=is_naive)} 파이 {expression_to_korean(expr.r, is_naive=is_naive)}"
    if isinstance(expr, RepeatedComb):
        return f"중복조합 {expression_to_korean(expr.n, is_naive=is_naive)} 에이치 {expression_to_korean(expr.r, is_naive=is_naive)}"
    if isinstance(expr, Log):
        if isinstance(expr.base, None_):
            return f"로그 {expression_to_korean(expr.arg, is_naive=is_naive)}"
        return f"로그 {expression_to_korean(expr.base, is_naive=is_naive)}의 {expression_to_korean(expr.arg, is_naive=is_naive)}"
    if isinstance(expr, SQRT):
        if str(expr.index) == "2":
            return f"루트 {expression_to_korean(expr.radicand, is_naive=is_naive)}"
        return f"{expression_to_korean(expr.index, is_naive=is_naive)} 제곱근 {expression_to_korean(expr.radicand, is_naive=is_naive)}"
    if isinstance(expr, SetBuilder):
        var = expression_to_korean(expr.var, is_naive=is_naive)
        condition = expression_to_korean(expr.condition, is_naive=is_naive).replace("이다", "일 때").replace("이고", "이며")
        return f"집합 {var} 바 {condition}"
    if isinstance(expr, LineExpr):
        line = expression_to_korean(expr.line, is_naive=is_naive)
        particles = get_particle(line)
        return f"직선 {line}{particles[0]} {expression_to_korean(expr.eq, is_naive=is_naive)}"
    if isinstance(expr, Ratio):
        return " 대 ".join(expression_to_korean(r, is_naive=is_naive) for r in expr.ratios)
    if isinstance(expr, Segment):
        return f"선분 {expression_to_korean(expr.start, is_naive=is_naive)} {expression_to_korean(expr.end, is_naive=is_naive)}"
    if isinstance(expr, Ray):
        return f"반직선 {expression_to_korean(expr.start, is_naive=is_naive)} {expression_to_korean(expr.through, is_naive=is_naive)}"
    if isinstance(expr, Line):
        return f"직선 {expression_to_korean(expr.point1, is_naive=is_naive)} {expression_to_korean(expr.point2, is_naive=is_naive)}"
    if isinstance(expr, Perp):
        line1 = expression_to_korean(expr.line1, is_naive=is_naive)
        particles1 = get_particle(line1)
        if is_descriptive_operator(expr.line2):
            line2_leftmost = expression_to_korean(get_leftmost_operand(expr.line2), is_naive=is_naive)
            particles2 = get_particle(line2_leftmost)
            line2_rest = expression_to_korean(expr.line2, is_nested=True, is_naive=is_naive)
            return f"{line1}{particles1[3]} {line2_leftmost}{particles2[0]} 수직이고 {line2_rest}"
        line2 = expression_to_korean(expr.line2, is_naive=is_naive)
        particles2 = get_particle(line2)
        if is_nested:
            return f"{line1}{particles1[3]} {line2}{particles2[0]} 수직이다"
        return f"{line1}{particles1[3]} {line2}{particles2[0]} 수직이다"

    
    if isinstance(expr, Paral):
        line1 = expression_to_korean(expr.line1, is_naive=is_naive)
        particles1 = get_particle(line1)
        if is_descriptive_operator(expr.line2):
            line2_leftmost = expression_to_korean(get_leftmost_operand(expr.line2), is_naive=is_naive)
            particles2 = get_particle(line2_leftmost)
            line2_rest = expression_to_korean(expr.line2, is_nested=True, is_naive=is_naive)
            return f"{line1}{particles1[3]} {line2_leftmost}{particles2[0]} 평행이고 {line2_rest}"
        line2 = expression_to_korean(expr.line2, is_naive=is_naive)
        particles2 = get_particle(line2)
        if is_nested:
            return f"{line1}{particles1[3]} {line2}{particles2[0]} 평행이다"
        return f"{line1}{particles1[3]} {line2}{particles2[0]} 평행이다"
    
    if isinstance(expr, InnerProduct):
        return f"{expression_to_korean(expr.vec1, is_naive=is_naive)} 내적 {expression_to_korean(expr.vec2, is_naive=is_naive)}"
    
    if isinstance(expr, Congru):
        shape1 = expression_to_korean(expr.shape1, is_naive=is_naive)
        particles1 = get_particle(shape1)
        if isinstance(expr.shape2, (Congru, Sim)):
            shape2_left = expression_to_korean(expr.shape2.shape1, is_naive=is_naive)
            particles2 = get_particle(shape2_left)
            shape2_rest = expression_to_korean(expr.shape2, is_nested=True, is_naive=is_naive)
            return f"{shape1}{particles1[3]} {shape2_left}{particles2[0]} 합동이고 {shape2_rest}"
        shape2 = expression_to_korean(expr.shape2, is_naive=is_naive)
        particles2 = get_particle(shape2)
        if is_nested:
            return f"{shape1}{particles1[3]} {shape2}{particles2[0]} 합동이다"
        return f"{shape1}{particles1[3]} {shape2}{particles2[0]} 합동이다"
    
    if isinstance(expr, Sim):
        shape1 = expression_to_korean(expr.shape1, is_naive=is_naive)
        particles1 = get_particle(shape1)
        if isinstance(expr.shape2, (Congru, Sim)):
            shape2_left = expression_to_korean(expr.shape2.shape1, is_naive=is_naive)
            particles2 = get_particle(shape2_left)
            shape2_rest = expression_to_korean(expr.shape2, is_nested=True, is_naive=is_naive)
            return f"{shape1}{particles1[3]} {shape2_left}{particles2[0]} 닮음이고 {shape2_rest}"
        shape2 = expression_to_korean(expr.shape2, is_naive=is_naive)
        particles2 = get_particle(shape2)
        if is_nested:
            return f"{shape1}{particles1[3]} {shape2}{particles2[0]} 닮음이다"
        return f"{shape1}{particles1[3]} {shape2}{particles2[0]} 닮음이다"
    
    if isinstance(expr, UnitDiv):
        return f"{expression_to_korean(expr.num_unit, is_naive=is_naive)} 퍼 {expression_to_korean(expr.denom_unit, is_naive=is_naive)}"
    if isinstance(expr, UnitMult):
        return f"{expression_to_korean(expr.left, is_naive=is_naive)} {expression_to_korean(expr.right, is_naive=is_naive)}"
    if isinstance(expr, About):
        return f"약 {expression_to_korean(expr.expr, is_naive=is_naive)}"
    
    if isinstance(expr, Point):
        if expr.name is not None and not isinstance(expr.name, None_):
            coords = " 콤마 ".join(expression_to_korean(arg, is_naive=is_naive) for arg in expr.args)
            name_korean = expression_to_korean(expr.name, is_naive=is_naive)
            if is_naive:
                return f"{name_korean} {coords}"
            else:
                return f"점 {name_korean} {coords}"
        coords = " 콤마 ".join(expression_to_korean(arg, is_naive=is_naive) for arg in expr.args)
        return coords
    
    # 3항 연산자
    if isinstance(expr, MixedFrac):
        whole = expression_to_korean(expr.whole, is_naive=is_naive)
        particles = get_particle(whole)
        return f"{whole}{particles[3]} {expression_to_korean(expr.denom, is_naive=is_naive)} 분의 {expression_to_korean(expr.num, is_naive=is_naive)}"
    
    if isinstance(expr, Diff):
        y = expression_to_korean(expr.y, is_naive=is_naive)
        x = expression_to_korean(expr.x, is_naive=is_naive)
        n_str = str(expr.n)
        if n_str == "1":
            return f"디 {y} 디 {x}"
        elif n_str == "2":
            return f"디 제곱 {y} 디 {x} 제곱"
        n = expression_to_korean(expr.n, is_naive=is_naive)
        return f"디 {n} 제곱 {y} 디 {x} {n} 제곱"
    
    # 가변 인자 연산자
    if isinstance(expr, Func):
        args_str = " 콤마 ".join(expression_to_korean(arg, is_naive=is_naive) for arg in expr.args)
        return f"{expression_to_korean(expr.name, is_naive=is_naive)} {args_str}"
    
    if isinstance(expr, FuncInv):
        args_str = " 콤마 ".join(expression_to_korean(arg, is_naive=is_naive) for arg in expr.args)
        return f"{expression_to_korean(expr.name, is_naive=is_naive)} 인버스 {args_str}"
    
    if isinstance(expr, SetRoster):
        elements_str = " 콤마 ".join(expression_to_korean(elem, is_naive=is_naive) for elem in expr.elements)
        return f"집합 {elements_str}"
    
    if isinstance(expr, Cases):
        cases_list = []
        for i, (case_expr, cond) in enumerate(expr.cases):
            expr_korean = expression_to_korean(case_expr, is_naive=is_naive)
            cond_korean = expression_to_korean(cond, is_naive=is_naive).replace("이다", "일 때").replace("이고", "이며")
            if i < len(expr.cases) - 1:
                cases_list.append(f"{cond_korean} {expr_korean} 그리고")
            else:
                cases_list.append(f"{cond_korean} {expr_korean}")
        return " ".join(cases_list)
    
    if isinstance(expr, Seq):
        term_korean = expression_to_korean(expr.term, is_naive=is_naive)
        return term_korean if is_naive else f"수열 {term_korean}"
    
    if isinstance(expr, Lim):
        var = expression_to_korean(expr.var, is_naive=is_naive)
        particles = get_particle(var)
        to_korean = expression_to_korean(expr.to, is_naive=is_naive)
        to_particles = get_particle(to_korean)
        return f"리미트 {var}{particles[1]} {to_korean}{to_particles[4]} 갈 때 {expression_to_korean(expr.expr, is_naive=is_naive)}"
    
    if isinstance(expr, Sum):
        var = expression_to_korean(expr.var, is_naive=is_naive)
        particles = get_particle(var)
        return f"시그마 {var}{particles[0]} {expression_to_korean(expr.start, is_naive=is_naive)}부터 {expression_to_korean(expr.end, is_naive=is_naive)}까지 {expression_to_korean(expr.term, is_naive=is_naive)}"
    
    if isinstance(expr, Integral):
        integrand = expression_to_korean(expr.integrand, is_naive=is_naive)
        var = expression_to_korean(expr.var, is_naive=is_naive)
        if isinstance(expr.lower, None_) and isinstance(expr.upper, None_):
            return f"인티그럴 {integrand} 디 {var}"
        elif isinstance(expr.lower, None_):
            return f"인티그럴 {expression_to_korean(expr.upper, is_naive=is_naive)}까지 {integrand} 디 {var}"
        elif isinstance(expr.upper, None_):
            return f"인티그럴 {expression_to_korean(expr.lower, is_naive=is_naive)}부터 {integrand} 디 {var}"
        return f"인티그럴 {expression_to_korean(expr.lower, is_naive=is_naive)}부터 {expression_to_korean(expr.upper, is_naive=is_naive)}까지 {integrand} 디 {var}"
    
    if isinstance(expr, Integrated):
        return f"{expression_to_korean(expr.antiderivative, is_naive=is_naive)} {expression_to_korean(expr.lower, is_naive=is_naive)}부터 {expression_to_korean(expr.upper, is_naive=is_naive)}까지"
    
    if isinstance(expr, Prob):
        event_korean = expression_to_korean(expr.event, is_naive=is_naive)
        if expr.condition is None or isinstance(expr.condition, None_):
            if is_naive:
                return f"피 {event_korean}"
            else:
                return f"확률 피 {event_korean}"
        else:
            condition_korean = expression_to_korean(expr.condition, is_naive=is_naive)
            if is_naive:
                return f"피 {event_korean} 바 {condition_korean}"
            else:
                return f"확률 피 {event_korean} 바 {condition_korean}"

    # ================== 구간 ==================
    if isinstance(expr, ClosedInterval):
        return f"구간 {expression_to_korean(expr.left, is_naive=is_naive)} 이상 {expression_to_korean(expr.right, is_naive=is_naive)} 이하"

    if isinstance(expr, ClosedOpenInterval):
        return f"구간 {expression_to_korean(expr.left, is_naive=is_naive)} 이상 {expression_to_korean(expr.right, is_naive=is_naive)} 미만"

    if isinstance(expr, OpenClosedInterval):
        return f"구간 {expression_to_korean(expr.left, is_naive=is_naive)} 초과 {expression_to_korean(expr.right, is_naive=is_naive)} 이하"

    if isinstance(expr, OpenInterval):
        return f"구간 {expression_to_korean(expr.left, is_naive=is_naive)} 초과 {expression_to_korean(expr.right, is_naive=is_naive)} 미만"
    
    # 삼각함수
    if isinstance(expr, Sin):
        return f"사인 {expression_to_korean(expr.arg, is_naive=is_naive)}"
    if isinstance(expr, Cos):
        return f"코사인 {expression_to_korean(expr.arg, is_naive=is_naive)}"
    if isinstance(expr, Tan):
        return f"탄젠트 {expression_to_korean(expr.arg, is_naive=is_naive)}"
    if isinstance(expr, Sec):
        return f"시컨트 {expression_to_korean(expr.arg, is_naive=is_naive)}"
    if isinstance(expr, Csc):
        return f"코시컨트 {expression_to_korean(expr.arg, is_naive=is_naive)}"
    if isinstance(expr, Cot):
        return f"코탄젠트 {expression_to_korean(expr.arg, is_naive=is_naive)}"
    if isinstance(expr, Ln):
        return f"엘엔 {expression_to_korean(expr.arg, is_naive=is_naive)}"     


    # 혼합 문장
    if isinstance(expr, InlinePhrase):
      return " ".join(expression_to_korean(p, is_naive=is_naive) for p in expr.parts)
    
    # 기본값
    return str(expr)

# ---------------- 공용 유틸 ----------------

# 실질 머리 토큰 선택: 조사/서술어/구두점 등은 건너뛰고 뒤에서부터 찾기
def pick_head_for_particle(tokens):
    if not tokens:
        return ""
    particle_set = {"은","는","이","가","을","를","와","과","으로","로"}
    bad = {
        "보다","이하","이상","이고","이며","이다","아니다",
        "의","콤마","분의","이면",
        "수직이다","평행이다","합동이다","닮음이다",
        "수직이고","평행이고","합동이고","닮음이고",
        "원소이다","원소이고","부분집합이다","포함한다","포함하지","않는다",
        "부터","까지","에서"
    }
    for tok, _ in reversed(tokens):
        if tok in particle_set or tok in bad:
            continue
        if tok.endswith("다") and tok not in {"제곱"}:
            continue
        if tok.strip() == "" or tok in {".",",","!","?"}:
            continue
        return tok
    return tokens[-1][0]


# --- 함수형(삼각함수/ln) 견고 처리 유틸 ---
TRIG_KOR = {
    "Sin": "사인", "Cos": "코사인", "Tan": "탄젠트",
    "Sec": "시컨트", "Csc": "코시컨트", "Cot": "코탄젠트",
    "sin": "사인", "cos": "코사인", "tan": "탄젠트",
    "sec": "시컨트", "csc": "코시컨트", "cot": "코탄젠트",
}
LN_NAMES = {"Ln": "엘엔", "ln": "엘엔"}

def _class_name(x):
    return getattr(x, "__class__", type(x)).__name__

def _get_func_name_from_Func(expr):
    # Func(...)일 때 함수 이름 문자열을 추출
    if not hasattr(expr, "name"):
        return None
    name = expr.name
    if isinstance(name, Value) and isinstance(name.val, str):
        return name.val
    if isinstance(name, Text):
        return name.text
    try:
        return str(name)
    except Exception:
        return None

def _get_unary_arg(expr):
    # 단항 함수의 피연산자 추출: Sin/Ln류 또는 Func(name)(arg)
    if hasattr(expr, "arg"):
        return expr.arg
    if hasattr(expr, "args") and getattr(expr, "args", None):
        return expr.args[0]
    return None

def _is_trig(expr):
    cname = _class_name(expr)
    if cname in TRIG_KOR:
        return True
    if cname == "Func":
        n = _get_func_name_from_Func(expr)
        return (n is not None) and (n.lower() in TRIG_KOR)
    return False

def _trig_kor(expr):
    cname = _class_name(expr)
    if cname in TRIG_KOR:
        return TRIG_KOR[cname]
    if cname == "Func":
        n = _get_func_name_from_Func(expr)
        if n:
            return TRIG_KOR.get(n.lower())
    return None

def _is_ln(expr):
    cname = _class_name(expr)
    if cname in LN_NAMES:
        return True
    if cname == "Func":
        n = _get_func_name_from_Func(expr)
        return (n is not None) and (n.lower() in LN_NAMES)
    return False

def _ln_kor(expr):
    cname = _class_name(expr)
    if cname in LN_NAMES:
        return LN_NAMES[cname]
    if cname == "Func":
        n = _get_func_name_from_Func(expr)
        if n:
            return LN_NAMES.get(n.lower())
    return None


# ---------------- 본 함수 ----------------

def expression_to_korean_with_depth(expr, depth=0, is_nested=False, is_naive=False):
    """
    Expression을 한글로 변환하면서 각 토큰의 depth 정보를 반환
    Returns: List[Tuple[str, int]]
    
    Args:
        expr: 변환할 Expression 객체
        depth: 현재 depth level
        is_nested: 중첩된 표현인지 여부
        is_naive: True일 경우 실무자 스타일로 형식적 수식어 생략
    """
    result = []  # [(token, depth), ...]

    # ---------- 0항 ----------
    if isinstance(expr, None_):
        return []

    if isinstance(expr, Value):
        val_str = str(expr.val)
        if val_str.replace('.', '').replace('-', '').isdigit():
            korean = number_to_korean(expr.val)
            return [(t, depth) for t in korean.split()]
        if len(val_str) == 1:
            ka = get_korean_alphabet(val_str)
            if ka.startswith("대문자 "):
                return [("대문자", depth), (ka[4:], depth)]
            return [(ka, depth)]
        tokens = []
        for c in val_str:
            if c.isalpha():
                ka = get_korean_alphabet(c)
                if ka.startswith("대문자 "):
                    tokens.append(("대문자", depth))
                    tokens.append((ka[4:], depth))
                else:
                    tokens.append((ka, depth))
            else:
                tokens.append((c, depth))
        return tokens

    if isinstance(expr, Text):
        return [(expr.text, depth)]

    if isinstance(expr, RecurringDecimal):
        result.extend(expression_to_korean_with_depth(expr.non_recurring, depth, is_naive=is_naive))
        result.append(("순환마디", depth))
        rec_digits = str(expr.recurring.val) if isinstance(expr.recurring, Value) else str(expr.recurring)
        rec_korean = "".join(number_to_korean(d).replace(" ", "") for d in rec_digits if d.isdigit())
        result.append((rec_korean, depth+1))
        return result

    if isinstance(expr, EmptySet): return [("공집합", depth)]
    if isinstance(expr, Infty):    return [("무한", depth)]
    if isinstance(expr, UpArrow):  return [("위화살표", depth)]
    if isinstance(expr, DownArrow):return [("아래화살표", depth)]
    if isinstance(expr, LeftArrow):return [("왼쪽화살표", depth)]
    if isinstance(expr, RightArrow):return [("오른쪽화살표", depth)]
    if isinstance(expr, Cdots):    return [("쩜쩜쩜", depth)]
    if isinstance(expr, Square):   return [("네모", depth)]
    if isinstance(expr, Circ):     return [("도", depth)]
    if isinstance(expr, BigCircle): return [("동그라미", depth)]
    if isinstance(expr, Degree):    return [("도", depth)]
    if isinstance(expr, EulerNum):
        if is_naive:
            return [("이", depth)]
        else:
            return [("자연상수", depth), ("이", depth)]    

    # ---------- 단항 ----------
    if isinstance(expr, Absolute):
        result.append(("절댓값", depth))
        result.extend(expression_to_korean_with_depth(expr.expr, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Factorial):
        result.extend(expression_to_korean_with_depth(expr.expr, depth+1, is_naive=is_naive))
        result.append(("팩토리얼", depth))
        return result

    if isinstance(expr, Plus):
        result.append(("플러스", depth))
        result.extend(expression_to_korean_with_depth(expr.expr, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Minus):
        result.append(("마이너스", depth))
        result.extend(expression_to_korean_with_depth(expr.expr, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Not):
        result.append(("낫", depth))
        result.extend(expression_to_korean_with_depth(expr.expr, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, SetComple):
        result.extend(expression_to_korean_with_depth(expr.expr, depth+1, is_naive=is_naive))
        result.append(("의", depth))
        result.append(("여집합", depth))
        return result

    if isinstance(expr, SetNum):
        result.extend(expression_to_korean_with_depth(expr.expr, depth+1, is_naive=is_naive))
        result.append(("의", depth))
        result.append(("원소의", depth))
        result.append(("개수", depth))
        return result

    if isinstance(expr, Delta):
        result.append(("델타", depth))
        result.extend(expression_to_korean_with_depth(expr.expr, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Bar):
        result.extend(expression_to_korean_with_depth(expr.expr, depth+1, is_naive=is_naive))
        result.append(("바", depth))
        return result

    if isinstance(expr, Vec):
        result.append(("벡터", depth))
        result.extend(expression_to_korean_with_depth(expr.expr, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Norm):
        result.append(("노름", depth))
        result.extend(expression_to_korean_with_depth(expr.expr, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Prime):
        # (expr)' → "expr 프라임"
        result.extend(expression_to_korean_with_depth(expr.expr, depth+1, is_naive=is_naive))
        result.append(("프라임", depth))
        return result

    if isinstance(expr, Gauss):
        # [expr] → "가우스 expr"
        result.append(("가우스", depth))
        result.extend(expression_to_korean_with_depth(expr.expr, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Unit):
        # Unit_dictionary에서 발음 찾기
        try:
            import csv
            import os
            dict_path = os.path.join(os.path.dirname(__file__), 'Unit_dictionary.csv')
            with open(dict_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['표기법'] == expr.unit:
                        result.append((row['한국어 발음'], depth))
                        return result
        except Exception as e:
            pass  # 파일이 없거나 오류가 발생하면 기본값 사용
        result.append((expr.unit, depth))  # 기본값
        return result

    if isinstance(expr, Triangle):
        return [("삼각형", depth), (expr.vertices, depth)]

    if isinstance(expr, Angle):
        return [("각", depth), (expr.vertices, depth)]

    if isinstance(expr, Arc):
        return [("호", depth), (expr.vertices, depth)]

    # --- 삼각함수/자연로그(견고화) ---
    if _is_trig(expr):
        kor = _trig_kor(expr)
        result.append((kor, depth))
        arg = _get_unary_arg(expr)
        result.extend(expression_to_korean_with_depth(arg, depth+1, is_naive=is_naive))
        return result

    if _is_ln(expr):
        kor = _ln_kor(expr)  # "엘엔"
        result.append((kor, depth))
        arg = _get_unary_arg(expr)
        result.extend(expression_to_korean_with_depth(arg, depth+1, is_naive=is_naive))
        return result

    # ---------- 2항 ----------
    if isinstance(expr, Add):
        if isinstance(expr.right, None_):
            result.extend(expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive))
            result.append(("플러스", depth))
        else:
            result.extend(expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive))
            result.append(("더하기", depth))
            result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Sub):
        if isinstance(expr.right, None_):
            result.extend(expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive))
            result.append(("마이너스", depth))
        else:
            result.extend(expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive))
            result.append(("빼기", depth))
            result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Mult):
        result.extend(expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive))
        result.append(("곱하기", depth))
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, ImplicitMult):
        result.extend(expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive))
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Divide):
        result.extend(expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive))
        result.append(("나누기", depth))
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, PlusMinus):
        result.extend(expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive))
        result.append(("플마", depth))
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, MinusPlus):
        result.extend(expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive))
        result.append(("마플", depth))
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Slash):
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        result.append(("분의", depth))
        result.extend(expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Frac):
        result.extend(expression_to_korean_with_depth(expr.denom, depth+1, is_naive=is_naive))
        result.append(("분의", depth))
        result.extend(expression_to_korean_with_depth(expr.num, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Power):
        # 삼각함수 거듭제곱(클래스/Func 모두 지원)
        if _is_trig(expr.base):
            trig_name = _trig_kor(expr.base)
            result.append((trig_name, depth+1))
            if str(expr.expo) == "2":
                result.append(("제곱", depth))
            else:
                result.extend(expression_to_korean_with_depth(expr.expo, depth+1, is_naive=is_naive))
                result.append(("제곱", depth))
            arg = _get_unary_arg(expr.base)
            result.extend(expression_to_korean_with_depth(arg, depth+2, is_naive=is_naive))
        else:
            result.extend(expression_to_korean_with_depth(expr.base, depth+1, is_naive=is_naive))
            if str(expr.expo) == "2":
                result.append(("제곱", depth))
            else:
                result.append(("의", depth))
                result.extend(expression_to_korean_with_depth(expr.expo, depth+1, is_naive=is_naive))
                result.append(("제곱", depth))
        return result

    # ---------- 비교 ----------
    if isinstance(expr, Less):
        left_tokens = expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive)
        result.extend(left_tokens)
        if left_tokens:
            head = pick_head_for_particle(left_tokens)
            result.append((get_particle(head)[1], depth))  # 이/가
        if is_descriptive_operator(expr.right):
            if is_descriptive_operator(expr.right.left if hasattr(expr.right,"left") else expr.right.elem):
                right_left_attr = expr.right.left if hasattr(expr.right,"left") else expr.right.elem
                right_left_tokens = expression_to_korean_with_depth(
                    right_left_attr.left if hasattr(right_left_attr,"left") else right_left_attr.elem, depth+1)
            else:
                right_left_tokens = expression_to_korean_with_depth(
                    expr.right.left if hasattr(expr.right,"left") else expr.right.elem, depth+1)
            result.extend(right_left_tokens)
            result.append(("보다", depth))
            result.append(("작고", depth))
            result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_nested=True, is_naive=is_naive))
        else:
            result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
            result.append(("보다", depth))
            result.append(("작다", depth))
        return result

    if isinstance(expr, Leq):
        left_tokens = expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive)
        result.extend(left_tokens)
        if left_tokens:
            head = pick_head_for_particle(left_tokens)
            result.append((get_particle(head)[1], depth))  # 이/가
        if is_descriptive_operator(expr.right):
            right_left_tokens = expression_to_korean_with_depth(
                expr.right.left if hasattr(expr.right,"left") else expr.right.elem, depth+2)
            result.extend(right_left_tokens)
            result.append(("이하", depth))
            result.append(("이고", depth))
            result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_nested=True, is_naive=is_naive))
        else:
            result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
            result.append(("이하", depth))
            result.append(("이다", depth))
        return result

    if isinstance(expr, Greater):
        left_tokens = expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive)
        result.extend(left_tokens)
        if left_tokens:
            head = pick_head_for_particle(left_tokens)
            result.append((get_particle(head)[1], depth))  # 이/가
        if is_descriptive_operator(expr.right):
            if is_descriptive_operator(expr.right.left if hasattr(expr.right,"left") else expr.right.elem):
                right_left_attr = expr.right.left if hasattr(expr.right,"left") else expr.right.elem
                right_left_tokens = expression_to_korean_with_depth(
                    right_left_attr.left if hasattr(right_left_attr,"left") else right_left_attr.elem, depth+3)
            else:
                right_left_tokens = expression_to_korean_with_depth(
                    expr.right.left if hasattr(expr.right,"left") else expr.right.elem, depth+2)
            result.extend(right_left_tokens)
            result.append(("보다", depth))
            result.append(("크고", depth))
            result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_nested=True, is_naive=is_naive))
        else:
            result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
            result.append(("보다", depth))
            result.append(("크다", depth))
        return result

    if isinstance(expr, Geq):
        left_tokens = expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive)
        result.extend(left_tokens)
        if left_tokens:
            head = pick_head_for_particle(left_tokens)
            result.append((get_particle(head)[1], depth))  # 이/가
        if is_descriptive_operator(expr.right):
            right_left_tokens = expression_to_korean_with_depth(
                expr.right.left if hasattr(expr.right,"left") else expr.right.elem, depth+2)
            result.extend(right_left_tokens)
            result.append(("이상", depth))
            result.append(("이고", depth))
            result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_nested=True, is_naive=is_naive))
        else:
            result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
            result.append(("이상", depth))
            result.append(("이다", depth))
        return result

    if isinstance(expr, Eq):
        left_tokens = expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive)
        result.extend(left_tokens)
        if left_tokens:
            head = pick_head_for_particle(left_tokens)
            result.append((get_particle(head)[0], depth))  # 은/는 (서술문)
        if is_nested and is_descriptive_operator(expr.right):
            right_left_attr = expr.right.left if hasattr(expr.right,"left") else expr.right.elem
            if is_descriptive_operator(right_left_attr):
                right_left_tokens = expression_to_korean_with_depth(
                    right_left_attr.left if hasattr(right_left_attr,"left") else right_left_attr.elem, depth+3)
            else:
                if isinstance(expr.right, (Less, Greater, Leq, Geq)):
                    right_left_tokens = expression_to_korean_with_depth(right_left_attr, depth+3, is_naive=is_naive)
                else:
                    right_left_tokens = expression_to_korean_with_depth(right_left_attr, depth+2, is_naive=is_naive)
            result.extend(right_left_tokens)
            result.append(("이고", depth))
            result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_nested=True, is_naive=is_naive))
        else:
            result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Neq):
        left_tokens = expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive)
        result.extend(left_tokens)
        if left_tokens:
            head = pick_head_for_particle(left_tokens)
            result.append((get_particle(head)[0], depth))  # 은/는
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        result.append(("이", depth))
        result.append(("아니다", depth))
        return result

    if isinstance(expr, Subscript):
        result.extend(expression_to_korean_with_depth(expr.base, depth+1, is_naive=is_naive))
        result.extend(expression_to_korean_with_depth(expr.sub, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Superscript):
        result.extend(expression_to_korean_with_depth(expr.base, depth+1, is_naive=is_naive))
        result.extend(expression_to_korean_with_depth(expr.sup, depth+1, is_naive=is_naive))
        return result

    # ---------- 집합 관계 ----------
    if isinstance(expr, SetIn):
        elem_tokens = expression_to_korean_with_depth(expr.elem, depth+1, is_naive=is_naive)
        result.extend(elem_tokens)
        if elem_tokens:
            head = pick_head_for_particle(elem_tokens)
            result.append((get_particle(head)[0], depth))  # 은/는
        if is_descriptive_operator(expr.set_expr):
            set_left_attr = expr.set_expr.elem if hasattr(expr.set_expr,"elem") else expr.set_expr.left
            if is_descriptive_operator(set_left_attr):
                set_left_tokens = expression_to_korean_with_depth(
                    set_left_attr.elem if hasattr(set_left_attr,"elem") else set_left_attr.left, depth+3)
            else:
                set_left_tokens = expression_to_korean_with_depth(set_left_attr, depth+1, is_naive=is_naive)
            result.extend(set_left_tokens)
            result.append(("의", depth))
            result.append(("원소이고", depth))
            result.extend(expression_to_korean_with_depth(expr.set_expr, depth+1, is_nested=True, is_naive=is_naive))
        else:
            result.extend(expression_to_korean_with_depth(expr.set_expr, depth+1, is_naive=is_naive))
            result.append(("의", depth))
            result.append(("원소이다", depth))
        return result

    if isinstance(expr, SetNotIn):
        elem_tokens = expression_to_korean_with_depth(expr.elem, depth+1, is_naive=is_naive)
        result.extend(elem_tokens)
        if elem_tokens:
            head = pick_head_for_particle(elem_tokens)
            result.append((get_particle(head)[0], depth))  # 은/는
        result.extend(expression_to_korean_with_depth(expr.set_expr, depth+1, is_naive=is_naive))
        result.append(("의", depth))
        result.append(("원소가", depth))
        result.append(("아니다", depth))
        return result

    if isinstance(expr, SetSub):
        left_tokens = expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive)
        result.extend(left_tokens)
        if left_tokens:
            head = pick_head_for_particle(left_tokens)
            result.append((get_particle(head)[0], depth))  # 은/는
        if is_descriptive_operator(expr.right):
            if is_descriptive_operator(expr.right.left):
                right_left_tokens = expression_to_korean_with_depth(
                    expr.right.left.left if hasattr(expr.right.left,"left") else expr.right.left.elem, depth+3)
            else:
                right_left_tokens = expression_to_korean_with_depth(expr.right.left, depth+2, is_naive=is_naive)
            result.extend(right_left_tokens)
            result.append(("의", depth))
            result.append(("부분집합이고", depth))
            result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_nested=True, is_naive=is_naive))
        else:
            result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
            result.append(("의", depth))
            result.append(("부분집합이다", depth))
        return result

    if isinstance(expr, SetNotSub):
        left_tokens = expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive)
        result.extend(left_tokens)
        if left_tokens:
            head = pick_head_for_particle(left_tokens)
            result.append((get_particle(head)[0], depth))  # 은/는
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        result.append(("의", depth))
        result.append(("부분집합이", depth))
        result.append(("아니다", depth))
        return result

    if isinstance(expr, SetSup):
        left_tokens = expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive)
        result.extend(left_tokens)
        if left_tokens:
            head = pick_head_for_particle(left_tokens)
            result.append((get_particle(head)[0], depth))  # 은/는
        if is_descriptive_operator(expr.right):
            if is_descriptive_operator(expr.right.left):
                right_left_tokens = expression_to_korean_with_depth(
                    expr.right.left.left if hasattr(expr.right.left,"left") else expr.right.left.elem, depth+3)
            else:
                right_left_tokens = expression_to_korean_with_depth(expr.right.left, depth+2, is_naive=is_naive)
            result.extend(right_left_tokens)
            result.append(("을", depth))
            result.append(("포함하고", depth))
            result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_nested=True, is_naive=is_naive))
        else:
            result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
            result.append(("을", depth))
            result.append(("포함한다", depth))
        return result

    if isinstance(expr, SetNotSup):
        left_tokens = expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive)
        result.extend(left_tokens)
        if left_tokens:
            head = pick_head_for_particle(left_tokens)
            result.append((get_particle(head)[0], depth))  # 은/는
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        result.append(("을", depth))
        result.append(("포함하지", depth))
        result.append(("않는다", depth))
        return result

    if isinstance(expr, SetCup):
        result.extend(expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive))
        result.append(("합집합", depth))
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, SetCap):
        result.extend(expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive))
        result.append(("교집합", depth))
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Rimpl):
        result.extend(expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive))
        result.append(("이면", depth))
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Limpl):
        left_tokens = expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive)
        result.extend(left_tokens)
        if left_tokens:
            head = pick_head_for_particle(left_tokens)
            result.append((get_particle(head)[0], depth))  # 은/는
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        result.append(("에", depth))
        result.append(("의해", depth))
        result.append(("함의된다", depth))
        return result

    if isinstance(expr, Biconditional):
        left_tokens = expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive)
        result.extend(left_tokens)
        if left_tokens:
            head = pick_head_for_particle(left_tokens)
            nen = get_particle(head)[0]
        else:
            nen = "는"
        result.append((nen, depth))
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        result.append(("의", depth))
        result.append(("필요충분조건이다", depth))
        return result

    if isinstance(expr, Iff):
        result.append(("참으로", depth))
        left_tokens = expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive)
        result.extend(left_tokens)
        if left_tokens:
            head = pick_head_for_particle(left_tokens)
            nen = get_particle(head)[0]
        else:
            nen = "는"
        result.append((nen, depth))
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        result.append(("의", depth))
        result.append(("필요충분조건이다", depth))
        return result

    if isinstance(expr, Rsufficient):
        left_tokens = expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive)
        result.extend(left_tokens)
        if left_tokens:
            head = pick_head_for_particle(left_tokens)
            result.append((get_particle(head)[0], depth))  # 은/는
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        result.append(("의", depth))
        result.append(("충분조건이다", depth))
        return result

    if isinstance(expr, Lsufficient):
        left_tokens = expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive)
        result.extend(left_tokens)
        if left_tokens:
            head = pick_head_for_particle(left_tokens)
            result.append((get_particle(head)[0], depth))  # 은/는
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        result.append(("의", depth))
        result.append(("필요조건이다", depth))
        return result

    if isinstance(expr, Prop):
        result.append(("명제", depth))
        result.extend(expression_to_korean_with_depth(expr.symbol, depth+1, is_naive=is_naive))
        result.extend(expression_to_korean_with_depth(expr.statement, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, FuncDef):
        result = []
        result.append(("함수", depth))

        func_tokens = expression_to_korean_with_depth(expr.func, depth + 1, is_naive=is_naive)
        result.extend(func_tokens)

        if func_tokens:
            head = pick_head_for_particle(func_tokens)
            result.append((get_particle(head)[0], depth))  # 은/는
        else:
            result.append(("는", depth))
        # 정의역: X
        domain_tokens = expression_to_korean_with_depth(expr.domain, depth + 1, is_naive=is_naive)
        result.extend(domain_tokens)

        result.append(("에서", depth))

        # 공역: Y
        codomain_tokens = expression_to_korean_with_depth(expr.codomain, depth + 1, is_naive=is_naive)
        result.extend(codomain_tokens)
        # codomain 뒤 '로/으로'
        if codomain_tokens:
            head = pick_head_for_particle(codomain_tokens)
            result.append((get_particle(head)[4], depth))  # 으로/로
        else:
            result.append(("으로", depth))

        # 마무리
        result.append(("가는", depth))
        result.append(("함수", depth))
        return result

    if isinstance(expr, Perm):
        result.append(("순열", depth))
        result.extend(expression_to_korean_with_depth(expr.n, depth+1, is_naive=is_naive))
        result.append(("피", depth))
        result.extend(expression_to_korean_with_depth(expr.r, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Comb):
        result.append(("조합", depth))
        result.extend(expression_to_korean_with_depth(expr.n, depth+1, is_naive=is_naive))
        result.append(("씨", depth))
        result.extend(expression_to_korean_with_depth(expr.r, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, RepeatedPermu):
        result.append(("중복순열", depth))
        result.extend(expression_to_korean_with_depth(expr.n, depth+1, is_naive=is_naive))
        result.append(("파이", depth))
        result.extend(expression_to_korean_with_depth(expr.r, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, RepeatedComb):
        result.append(("중복조합", depth))
        result.extend(expression_to_korean_with_depth(expr.n, depth+1, is_naive=is_naive))
        result.append(("에이치", depth))
        result.extend(expression_to_korean_with_depth(expr.r, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Log):
        result.append(("로그", depth))
        if isinstance(expr.base, None_):
            result.extend(expression_to_korean_with_depth(expr.arg, depth+1, is_naive=is_naive))
        else:
            result.extend(expression_to_korean_with_depth(expr.base, depth+1, is_naive=is_naive))
            result.append(("의", depth))
            result.extend(expression_to_korean_with_depth(expr.arg, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, SQRT):
        if str(expr.index) == "2":
            result.append(("루트", depth))
            result.extend(expression_to_korean_with_depth(expr.radicand, depth+1, is_naive=is_naive))
        else:
            result.extend(expression_to_korean_with_depth(expr.index, depth+1, is_naive=is_naive))
            result.append(("제곱근", depth))
            result.extend(expression_to_korean_with_depth(expr.radicand, depth+1, is_naive=is_naive))
        return result

    # ---------- SetBuilder (조건제시법) ----------
    if isinstance(expr, SetBuilder):
        result.append(("집합", depth))
        result.extend(expression_to_korean_with_depth(expr.var, depth+1, is_naive=is_naive))
        result.append(("바", depth))

        cond = expr.condition

        # InlinePhrase: 텍스트/수식 자유 혼합
        if isinstance(cond, InlinePhrase):
            for part in cond.parts:
                if isinstance(part, Text):
                    text = re.sub(r'([은는이가을를])(있다|없다)', r'\1 \2', part.text)
                    text = re.sub(r'[.,!?]+', ' ', text)
                    for w in filter(None, text.split()):
                        result.append((w, depth+1))
                elif (isinstance(part, Value) and isinstance(part.val, str)
                      and len(part.val) == 1 and part.val.isalpha()):
                    result.append((part.val, depth+1))  # 원문 단일 영문자
                else:
                    result.extend(expression_to_korean_with_depth(part, depth+1, is_naive=is_naive))
            return result

        # 일반 조건 수식
        result.extend(expression_to_korean_with_depth(cond, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, LineExpr):
        line_tokens = expression_to_korean_with_depth(expr.line, depth+1, is_naive=is_naive)
        result.append(("직선", depth))
        result.extend(line_tokens)
        if line_tokens:
            head = pick_head_for_particle(line_tokens)
            result.append((get_particle(head)[0], depth))  # 은/는
        result.extend(expression_to_korean_with_depth(expr.eq, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Ratio):
        for i, r in enumerate(expr.ratios):
            if i > 0:
                result.append(("대", depth))
            result.extend(expression_to_korean_with_depth(r, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Segment):
        result.append(("선분", depth))
        result.extend(expression_to_korean_with_depth(expr.start, depth+1, is_naive=is_naive))
        result.extend(expression_to_korean_with_depth(expr.end, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Ray):
        result.append(("반직선", depth))
        result.extend(expression_to_korean_with_depth(expr.start, depth+1, is_naive=is_naive))
        result.extend(expression_to_korean_with_depth(expr.through, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Line):
        result.append(("직선", depth))
        result.extend(expression_to_korean_with_depth(expr.point1, depth+1, is_naive=is_naive))
        result.extend(expression_to_korean_with_depth(expr.point2, depth+1, is_naive=is_naive))
        return result

    # ---------- 기하: 수직/평행 ----------
    if isinstance(expr, Perp):
        result = []
        line1_t = expression_to_korean_with_depth(expr.line1, depth+1, is_naive=is_naive)
        result.extend(line1_t)
        head1 = pick_head_for_particle(line1_t)
        result.append((get_particle(head1)[3], depth))  # 와/과
        if is_descriptive_operator(expr.line2):
            leftmost = get_leftmost_operand(expr.line2)
            left2_t = expression_to_korean_with_depth(leftmost, depth+2, is_naive=is_naive)
            result.extend(left2_t)
            head2 = pick_head_for_particle(left2_t)
            result.append((get_particle(head2)[0], depth))  # 은/는
            result.append(("수직이고", depth))
            result.extend(expression_to_korean_with_depth(expr.line2, depth+1, is_nested=True, is_naive=is_naive))
        else:
            line2_t = expression_to_korean_with_depth(expr.line2, depth+1, is_naive=is_naive)
            result.extend(line2_t)
            head2 = pick_head_for_particle(line2_t)
            result.append((get_particle(head2)[0], depth))  # 은/는
            result.append(("수직이다", depth))
        return result

    if isinstance(expr, Paral):
        result = []
        line1_t = expression_to_korean_with_depth(expr.line1, depth+1, is_naive=is_naive)
        result.extend(line1_t)
        head1 = pick_head_for_particle(line1_t)
        result.append((get_particle(head1)[3], depth))  # 와/과
        if is_descriptive_operator(expr.line2):
            leftmost = get_leftmost_operand(expr.line2)
            left2_t = expression_to_korean_with_depth(leftmost, depth+2, is_naive=is_naive)
            result.extend(left2_t)
            head2 = pick_head_for_particle(left2_t)
            result.append((get_particle(head2)[0], depth))  # 은/는
            result.append(("평행이고", depth))
            result.extend(expression_to_korean_with_depth(expr.line2, depth+1, is_nested=True, is_naive=is_naive))
        else:
            line2_t = expression_to_korean_with_depth(expr.line2, depth+1, is_naive=is_naive)
            result.extend(line2_t)
            head2 = pick_head_for_particle(line2_t)
            result.append((get_particle(head2)[0], depth))  # 은/는
            result.append(("평행이다", depth))
        return result

    if isinstance(expr, InnerProduct):
        result.extend(expression_to_korean_with_depth(expr.vec1, depth+1, is_naive=is_naive))
        result.append(("내적", depth))
        result.extend(expression_to_korean_with_depth(expr.vec2, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Congru):
        shape1_tokens = expression_to_korean_with_depth(expr.shape1, depth+1, is_naive=is_naive)
        result.extend(shape1_tokens)
        if shape1_tokens:
            head = pick_head_for_particle(shape1_tokens)
            result.append((get_particle(head)[3], depth))  # 와/과
        if isinstance(expr.shape2, (Congru, Sim)):
            if is_descriptive_operator(expr.shape2.shape1):
                shape2_left_tokens = expression_to_korean_with_depth(
                    expr.shape2.shape1.shape1 if hasattr(expr.shape2.shape1,"shape1") else expr.shape2.shape1.elem, depth+3)
            else:
                shape2_left_tokens = expression_to_korean_with_depth(expr.shape2.shape1, depth+2, is_naive=is_naive)
            result.extend(shape2_left_tokens)
            if shape2_left_tokens:
                head = pick_head_for_particle(shape2_left_tokens)
                result.append((get_particle(head)[0], depth))  # 은/는
            result.append(("합동이고", depth))
            result.extend(expression_to_korean_with_depth(expr.shape2, depth+1, is_nested=True, is_naive=is_naive))
        else:
            shape2_tokens = expression_to_korean_with_depth(expr.shape2, depth+1, is_naive=is_naive)
            result.extend(shape2_tokens)
            if shape2_tokens:
                head = pick_head_for_particle(shape2_tokens)
                result.append((get_particle(head)[0], depth))  # 은/는
            result.append(("합동이다", depth))
        return result

    if isinstance(expr, Sim):
        shape1_tokens = expression_to_korean_with_depth(expr.shape1, depth+1, is_naive=is_naive)
        result.extend(shape1_tokens)
        if shape1_tokens:
            head = pick_head_for_particle(shape1_tokens)
            result.append((get_particle(head)[3], depth))  # 와/과
        if isinstance(expr.shape2, (Congru, Sim)):
            if is_descriptive_operator(expr.shape2.shape1):
                shape2_left_tokens = expression_to_korean_with_depth(
                    expr.shape2.shape1.shape1 if hasattr(expr.shape2.shape1,"shape1") else expr.shape2.shape1.elem, depth+3)
            else:
                shape2_left_tokens = expression_to_korean_with_depth(expr.shape2.shape1, depth+2, is_naive=is_naive)
            result.extend(shape2_left_tokens)
            if shape2_left_tokens:
                head = pick_head_for_particle(shape2_left_tokens)
                result.append((get_particle(head)[0], depth))  # 은/는
            result.append(("닮음이고", depth))
            result.extend(expression_to_korean_with_depth(expr.shape2, depth+1, is_nested=True, is_naive=is_naive))
        else:
            shape2_tokens = expression_to_korean_with_depth(expr.shape2, depth+1, is_naive=is_naive)
            result.extend(shape2_tokens)
            if shape2_tokens:
                head = pick_head_for_particle(shape2_tokens)
                result.append((get_particle(head)[0], depth))  # 은/는
            result.append(("닮음이다", depth))
        return result

    # ================== 구간 ==================
    if isinstance(expr, ClosedInterval):
        result.extend(expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive))
        result.append(("이상", depth))
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        result.append(("이하", depth))        
        return result

    if isinstance(expr, ClosedOpenInterval):
        result.extend(expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive))
        result.append(("이상", depth))
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        result.append(("미만", depth))        
        return result

    if isinstance(expr, OpenClosedInterval):
        result.extend(expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive))
        result.append(("초과", depth))
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        result.append(("이하", depth))        
        return result

    if isinstance(expr, OpenInterval):
        result.extend(expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive))
        result.append(("초과", depth))
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        result.append(("미만", depth))        
        return result

    # ---------- 단위/기타 ----------
    if isinstance(expr, UnitDiv):
        result.extend(expression_to_korean_with_depth(expr.num_unit, depth+1, is_naive=is_naive))
        result.append(("퍼", depth))
        result.extend(expression_to_korean_with_depth(expr.denom_unit, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, UnitMult):
        result.extend(expression_to_korean_with_depth(expr.left, depth+1, is_naive=is_naive))
        result.extend(expression_to_korean_with_depth(expr.right, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, About):
        result.append(("약", depth))
        result.extend(expression_to_korean_with_depth(expr.expr, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Point):
        if expr.name is not None and not isinstance(expr.name, None_):
            if not is_naive:
                result.append(("점", depth))
            result.extend(expression_to_korean_with_depth(expr.name, depth+1, is_naive=is_naive))
            for i, arg in enumerate(expr.args):
                result.extend(expression_to_korean_with_depth(arg, depth+1, is_naive=is_naive))
                if i < len(expr.args) - 1:
                    result.append(("콤마", depth))
        else:
            for i, arg in enumerate(expr.args):
                result.extend(expression_to_korean_with_depth(arg, depth+1, is_naive=is_naive))
                if i < len(expr.args) - 1:
                    result.append(("콤마", depth))
        return result

    # ---------- 3항 ----------
    if isinstance(expr, MixedFrac):
        whole_tokens = expression_to_korean_with_depth(expr.whole, depth+1, is_naive=is_naive)
        result.extend(whole_tokens)
        if whole_tokens:
            head = pick_head_for_particle(whole_tokens)
            result.append((get_particle(head)[3], depth))  # 와/과
        result.extend(expression_to_korean_with_depth(expr.denom, depth+1, is_naive=is_naive))
        result.append(("분의", depth))
        result.extend(expression_to_korean_with_depth(expr.num, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Diff):
        n_str = str(expr.n)
        result.append(("디", depth))
        if n_str != "1":
            if n_str == "2":
                result.append(("제곱", depth+1))
            else:
                result.extend(expression_to_korean_with_depth(expr.n, depth+1, is_naive=is_naive))
                result.append(("제곱", depth+1))
        result.extend(expression_to_korean_with_depth(expr.y, depth+1, is_naive=is_naive))
        result.append(("디", depth))
        result.extend(expression_to_korean_with_depth(expr.x, depth+1, is_naive=is_naive))
        if n_str != "1":
            if n_str == "2":
                result.append(("제곱", depth+1))
            else:
                result.extend(expression_to_korean_with_depth(expr.n, depth+1, is_naive=is_naive))
                result.append(("제곱", depth+1))
        return result

    # ---------- 가변 인자 ----------
    if isinstance(expr, Func):
        result.extend(expression_to_korean_with_depth(expr.name, depth+1, is_naive=is_naive))
        for i, arg in enumerate(expr.args):
            result.extend(expression_to_korean_with_depth(arg, depth+1, is_naive=is_naive))
            if i < len(expr.args) - 1:
                result.append(("콤마", depth))
        return result

    if isinstance(expr, FuncInv):
        result.extend(expression_to_korean_with_depth(expr.name, depth+1, is_naive=is_naive))
        result.append(("인버스", depth))
        for i, arg in enumerate(expr.args):
            result.extend(expression_to_korean_with_depth(arg, depth+1, is_naive=is_naive))
            if i < len(expr.args) - 1:
                result.append(("콤마", depth))
        return result

    if isinstance(expr, SetRoster):
        result.append(("집합", depth))
        for i, elem in enumerate(expr.elements):
            result.extend(expression_to_korean_with_depth(elem, depth+1, is_naive=is_naive))
            if i < len(expr.elements) - 1:
                result.append(("콤마", depth))
        return result

    if isinstance(expr, Cases):
        for i, (case_expr, cond) in enumerate(expr.cases):
            cond_tokens = expression_to_korean_with_depth(cond, depth+1, is_nested=False, is_naive=is_naive)
            modified = []
            for token, d in cond_tokens:
                if token == "이다":
                    modified.append(("일", d)); modified.append(("때", d))
                elif token == "이고":
                    modified.append(("이며", d))
                else:
                    modified.append((token, d))
            result.extend(modified)
            result.extend(expression_to_korean_with_depth(case_expr, depth+1, is_naive=is_naive))
            if i < len(expr.cases) - 1:
                result.append(("그리고", depth))
        return result

    if isinstance(expr, Seq):
        if not is_naive:
            result.append(("수열", depth))
        result.extend(expression_to_korean_with_depth(expr.term, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Lim):
        var_tokens = expression_to_korean_with_depth(expr.var, depth+1, is_naive=is_naive)
        to_tokens  = expression_to_korean_with_depth(expr.to,  depth+1, is_naive=is_naive)
        result.append(("리미트", depth))
        result.extend(var_tokens)
        if var_tokens:
            head = pick_head_for_particle(var_tokens)
            result.append((get_particle(head)[1], depth))  # 이/가
        result.extend(to_tokens)
        if to_tokens:
            head = pick_head_for_particle(to_tokens)
            result.append((get_particle(head)[4], depth))  # 으로/로
        result.append(("갈", depth))
        result.append(("때", depth))
        result.extend(expression_to_korean_with_depth(expr.expr, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Sum):
        var_tokens = expression_to_korean_with_depth(expr.var, depth+1, is_naive=is_naive)
        result.append(("시그마", depth))
        result.extend(var_tokens)
        if var_tokens:
            head = pick_head_for_particle(var_tokens)
            result.append((get_particle(head)[0], depth))  # 은/는
        result.extend(expression_to_korean_with_depth(expr.start, depth+1, is_naive=is_naive))
        result.append(("부터", depth))
        result.extend(expression_to_korean_with_depth(expr.end, depth+1, is_naive=is_naive))
        result.append(("까지", depth))
        result.extend(expression_to_korean_with_depth(expr.term, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Integral):
        lower_none = isinstance(expr.lower, None_)
        upper_none = isinstance(expr.upper, None_)
        result.append(("인티그럴", depth))
        if not lower_none and not upper_none:
            result.extend(expression_to_korean_with_depth(expr.lower, depth+1, is_naive=is_naive))
            result.append(("부터", depth))
            result.extend(expression_to_korean_with_depth(expr.upper, depth+1, is_naive=is_naive))
            result.append(("까지", depth))
        elif not upper_none:
            result.extend(expression_to_korean_with_depth(expr.upper, depth+1, is_naive=is_naive))
            result.append(("까지", depth))
        elif not lower_none:
            result.extend(expression_to_korean_with_depth(expr.lower, depth+1, is_naive=is_naive))
            result.append(("부터", depth))
        result.extend(expression_to_korean_with_depth(expr.integrand, depth+1, is_naive=is_naive))
        result.append(("디", depth))
        result.extend(expression_to_korean_with_depth(expr.var, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, Integrated):
        result.extend(expression_to_korean_with_depth(expr.antiderivative, depth+1, is_naive=is_naive))
        result.extend(expression_to_korean_with_depth(expr.lower, depth+1, is_naive=is_naive))
        result.append(("부터", depth))
        result.extend(expression_to_korean_with_depth(expr.upper, depth+1, is_naive=is_naive))
        result.append(("까지", depth))
        return result

    if isinstance(expr, Prob):
        if not is_naive:
            result.append(("확률", depth))
        result.append(("피", depth))
        result.extend(expression_to_korean_with_depth(expr.event, depth+1, is_naive=is_naive))
        if expr.condition is not None and not isinstance(expr.condition, None_):
            result.append(("바", depth))
            result.extend(expression_to_korean_with_depth(expr.condition, depth+1, is_naive=is_naive))
        return result

    if isinstance(expr, InlinePhrase):
        for part in expr.parts:
            if isinstance(part, Text):
                # 붙임표기 분리: "가있다" → "가 있다"
                text = re.sub(r'([은는이가을를])(있다|없다)', r'\1 \2', part.text)
                # 문장부호 제거
                text = re.sub(r'[.,!?]+', ' ', text)
                # 공백 단위 토큰화 (최상위이므로 depth 그대로)
                for w in filter(None, text.split()):
                    result.append((w, depth))

            elif (isinstance(part, Value) and isinstance(part.val, str)
                  and len(part.val) == 1 and part.val.isalpha()):
                # 단일 영문자(Value("x"))는 원문 유지
                result.append((part.val, depth))

            else:
                # 숫자/수식 등은 기존 규칙으로 재귀 (depth 그대로)
                result.extend(expression_to_korean_with_depth(part, depth, is_naive=is_naive))
        return result


    # ---------- 기본값 ----------
    return [(str(expr), depth)]

def pretty_print_korean_and_depth(expr, description, answer_data=None):
    """
    Expression을 깔끔하게 출력하고 정답과 비교
    """
    print("=" * 100, description, "=" * 100)

    # 기본 정보
    print(f"수식: {str(expr)}")
    print(f"Repr: {repr(expr)}")

    # 한글 변환 + depth
    tokens_with_depth = expression_to_korean_with_depth(expr)
    korean_text = " ".join(token for token, _ in tokens_with_depth)

    print(f"대체 텍스트: {expression_to_korean(expr)}")

    # 토큰+깊이 출력
    print("토큰+깊이:")
    print(f"  텍스트: {korean_text}")
    for token, depth in tokens_with_depth:
        print(f"  {token}: {depth}")

    # 정답과 비교
    if answer_data:
        # 텍스트 일치 여부

        s_norm = expression_to_korean(expr).replace(" ", "")
        t_norm = korean_text.replace(" ", "")
        text_match = "O" if s_norm == t_norm else "X"
        print(f"일치(O/X): {'O' if text_match else 'X'}")

        # depth 일치 여부
        predicted_tokens = [{"token": t, "depth": d} for t, d in tokens_with_depth]
        correct_tokens = answer_data["tokens_with_depth"]

        depth_match = (predicted_tokens == correct_tokens)
        print(f"correct depth(O/X): {'O' if depth_match else 'X'}")

        if not depth_match:
            # 불일치 원인 파악
            if len(t_norm) != len(s_norm):
                print(f"└─ 길이 다름 (예측 {len(predicted_tokens)} vs 정답 {len(correct_tokens)})")
            else:
                # 어디가 다른지 찾기
                for i, (pred, corr) in enumerate(zip(predicted_tokens, correct_tokens)):
                    if pred != corr:
                        print(f"└─ 위치 {i}: 예측 {pred} vs 정답 {corr}")
                        break

            # 정답 출력
            print("\n[정답 토큰+깊이]")
            for item in correct_tokens:
                print(f"  {item['token']}: {item['depth']}")
    visualize_expression_tree(expr)
    print()

# ===================== Helper Functions =====================
def pick_head_for_particle(korean_tokens: List[Tuple[str, int, int]]) -> str:
    """
    조사 붙일 때 기준이 되는 '머리 단어'를 뒤에서부터 골라준다.
    korean_tokens: [(token_str, pitch, volume), ...]
    """
    if not korean_tokens:
        return ""

    particle_set = {"은", "는", "이", "가", "을", "를", "와", "과", "으로", "로"}
    # 조사/서술어/접속사 등, 머리가 되기 애매한 것들
    bad = {
        "보다", "이하이다", "이상이다", "작다", "크다",
        "이고", "이며", "이다", "아니다",
        "의", "콤마", "분의", "이면",
        "수직이다", "평행이다", "합동이다", "닮음이다",
        "수직이고", "평행이고", "합동이고", "닮음이고",
        "원소이다", "원소가", "부분집합이다", "포함한다", "포함하지", "않는다",
        "부터", "까지", "에서"
    }

    for tok, *_ in reversed(korean_tokens):
        if tok in particle_set or tok in bad:
            continue
        if tok.strip() == "" or tok in {".", ",", "!", "?"}:
            continue
        return tok

    # 다 걸러져도 그냥 마지막 토큰이라도 사용
    return korean_tokens[-1][0]

def pick_head_for_particle_from_tokens(tokens):
    """
    토큰 리스트에서 조사 판단용 '마지막 단어'를 고른다.
    tokens: [(word, pitch, volume), ...]
    """
    for w, p, v in reversed(tokens):
        return w
    return ""


def class_name(e):
    """객체의 클래스 이름 반환"""
    return getattr(e, "__class__", type(e)).__name__


def is_operator(e):
    """연산자인지 판별"""
    return isinstance(e, (
        Add, Sub, Mult, ImplicitMult, Divide, Slash, Frac, MixedFrac,
        Plus, Minus, PlusMinus, MinusPlus,
        Power, SQRT, Absolute, Gauss,
        Eq, Neq, Less, Leq, Greater, Geq,
        SetIn, SetNotIn, SetBuilder, SetRoster,
        SetSub, SetSup, SetNotSub, SetNotSup,
        SetCup, SetCap, SetNum, SetComple,
        Subscript, Superscript,
        Not, Rimpl, Limpl, Biconditional, Iff, Prop,
        Rsufficient, Lsufficient,
        FuncDef, Func, FuncInv,
        Perm, Comb, RepeatedPermu, RepeatedComb,
        Factorial,
        Log, Ln, Sin, Cos, Tan, Sec, Csc, Cot,
        Seq, Lim, Sum, Delta, Prime, Diff,
        Integral, Integrated,
        Prob, Bar,
        LineExpr,
        Segment, Ray, Line, Vec,
        Perp, Paral, Ratio,
        Point, Triangle, Angle, Norm, Arc,
        InnerProduct, Congru, Sim
    ))


def is_infix(e):
    """중위 연산자인지 판별"""
    return isinstance(e, (
        Add, Sub, Mult, ImplicitMult, Divide, Slash, Frac, MixedFrac,
        PlusMinus, MinusPlus,
        Eq, Neq, Less, Leq, Greater, Geq,
        SetIn, SetNotIn,
        SetSub, SetSup, SetNotSub, SetNotSup,
        SetCup, SetCap,
        Rimpl, Limpl, Biconditional, Iff,
        Rsufficient, Lsufficient,
        InnerProduct
    ))


def is_postfix(e):
    """후위 연산자인지 판별"""
    return isinstance(e, (
        Power, SetComple, Factorial, Prime,
        Bar, Integrated, FuncInv
    ))


def is_mid_or_postfix(e):
    """중후위 연산자인지 판별"""
    return is_infix(e) or is_postfix(e)

def is_add_like(e):
    """Add, Sub, PlusMinus, MinusPlus 계열인지 판별"""
    return isinstance(e, (Add, Sub, PlusMinus, MinusPlus))


def is_mult_chain(e):
    """곱/나눗셈 계열인지 판별"""
    return isinstance(e, (Divide, Mult, ImplicitMult, InnerProduct))


def is_add_mult_chain(e):
    """덧셈/뺄셈/곱셈/나눗셈 계열인지 판별"""
    return isinstance(e, (Add, Sub, PlusMinus, MinusPlus, Divide, Mult, ImplicitMult))

def is_integer_value(expr):
    """정수 Value인지 판별"""
    if not isinstance(expr, Value):
        return False
    s = str(expr.val)
    return s.replace('-', '').replace('+', '').isdigit()

# ===================== Main Function =====================

def expression_to_tokens_with_pitch(expr, d=0, in_condition=False, is_naive=False) -> List[Tuple[str, int, int]]:
    """
    Expression을 [(token, pitch_level, volume_level), ...] 로 변환
    
    Args:
        expr: Expression 객체
        d: pitch level (depth)
        in_condition: 조건부 표현 안에 있는지 여부
        is_naive: True일 경우 실무자 스타일로 형식적 수식어 생략
        d: 현재 pitch level (기본값 0)
        in_condition: 조건부 맥락(Prob의 event/condition, SetBuilder의 condition 등)에서 호출되는지 여부
    
    Returns:
        List of (token_string, pitch_level, volume_level) tuples
    """
    tokens = []

    # ==================== 0항 / 비연산자 ====================

    if isinstance(expr, None_):
        return []

    if isinstance(expr, Value):
        val = expr.val
        s = str(val)
        # 숫자 처리
        if s.replace('.', '').replace('-', '').replace('+', '').isdigit() or '.' in s:
            korean = number_to_korean(val)
            # 공백으로 분리된 단어들을 개별 토큰으로
            for word in korean.split():
                tokens.append((word, d, 0))
            return tokens
        # 단일 알파벳
        if len(s) == 1 and s.isalpha():
            korean = get_korean_alphabet(s)
            return [(korean, d, 0)]
        # 복합 문자열
        result = []
        for c in s:
            if c.isalpha():
                result.append((get_korean_alphabet(c), d, 0))
            elif c.isspace():
                continue
            else:
                result.append((c, d, 0))
        return result

    if isinstance(expr, Text):
        # 공백으로 분리하여 개별 토큰으로
        words = expr.text.split()
        return [(w, d, 0) for w in words if w]

    if isinstance(expr, RecurringDecimal):
        # a의 문자열 확인
        non_rec_str = str(expr.non_recurring)
        has_dot = '.' in non_rec_str
        
        a_tokens = expression_to_tokens_with_pitch(expr.non_recurring, d, in_condition, is_naive)
        b_tokens = expression_to_tokens_with_pitch(expr.recurring, d, in_condition, is_naive)
        
        tokens.extend(a_tokens)
        if not has_dot:
            tokens.append(("점", d, 0))
        tokens.append(("순환마디", d, 0))
        tokens.extend(b_tokens)
        return tokens

    if isinstance(expr, EmptySet):
        return [("공집합", d, 0)]

    if isinstance(expr, Infty):
        return [("무한", d, 0)]

    if isinstance(expr, UpArrow):
        return [("위화살표", d, 0)]

    if isinstance(expr, DownArrow):
        return [("아래화살표", d, 0)]

    if isinstance(expr, LeftArrow):
        return [("왼쪽화살표", d, 0)]

    if isinstance(expr, RightArrow):
        return [("오른쪽화살표", d, 0)]

    if isinstance(expr, Cdots):
        return [("쩜쩜쩜", d, 0)]

    if isinstance(expr, Square):
        return [("네모", d, 0)]

    if isinstance(expr, Circ):
        return [("도", d, 0)]

    if isinstance(expr, BigCircle):
        return [("동그라미", d, 0)]

    if isinstance(expr, Degree):
        return [("도", d, 0)]

    if isinstance(expr, EulerNum):
        if is_naive:
            return [("이", d, 0)]
        else:
            return [("자연상수", d, 0), ("이", d, 0)]

    # ==================== 단항 연산자 ====================

    if isinstance(expr, Absolute):
        x = expr.expr
        x_d = d + 1 if is_mid_or_postfix(x) else d
        tokens.append(("절댓값", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(x, x_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Gauss):
        x = expr.expr
        x_d = d + 1 if is_mid_or_postfix(x) else d
        tokens.append(("가우스", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(x, x_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Plus):
        a = expr.expr
        a_d = d + 1 if is_mid_or_postfix(a) else d
        tokens.append(("플러스", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(a, a_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Minus):
        a = expr.expr
        a_d = d + 1 if is_mid_or_postfix(a) else d
        tokens.append(("마이너스", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(a, a_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Not):
        p = expr.expr
        p_d = d + 1 if is_mid_or_postfix(p) else d
        tokens.append(("낫", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(p, p_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Delta):
        x = expr.expr
        x_d = d + 1 if is_mid_or_postfix(x) else d
        tokens.append(("델타", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(x, x_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Vec):
        a = expr.expr
        tokens.append(("벡터", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(a, d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Norm):
        v = expr.expr
        v_d = d + 1 if is_mid_or_postfix(v) else d
        tokens.append(("노름", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(v, v_d, in_condition, is_naive))
        return tokens

    # ==================== 후위 연산자 ====================

    if isinstance(expr, Factorial):
        n = expr.expr
        n_d = d + 1 if is_operator(n) else d
        tokens.extend(expression_to_tokens_with_pitch(n, n_d, in_condition, is_naive))
        tokens.append(("팩토리얼", d, 0))
        return tokens

    if isinstance(expr, Prime):
        y = expr.expr
        # y가 연산자이고 Prime이 아니면 d+1
        if is_operator(y) and not isinstance(y, Prime):
            y_d = d + 1
        else:
            y_d = d
        tokens.extend(expression_to_tokens_with_pitch(y, y_d, in_condition, is_naive))
        tokens.append(("프라임", d, 0))
        return tokens

    if isinstance(expr, SetComple):
        A = expr.expr
        A_d = d + 1 if is_operator(A) else d
        tokens.extend(expression_to_tokens_with_pitch(A, A_d, in_condition, is_naive))
        tokens.append(("의", d, 0))
        tokens.append(("여집합", d, 0))
        return tokens

    if isinstance(expr, Bar):
        X = expr.expr
        X_d = d + 1 if is_operator(X) else d
        tokens.extend(expression_to_tokens_with_pitch(X, X_d, in_condition, is_naive))
        tokens.append(("바", d, 0))
        return tokens

    # ==================== 2항 연산자: 덧셈/뺄셈 ====================

    if isinstance(expr, Add):
        a, b = expr.left, expr.right
        # None_인지 확인 (우극한/좌극한)
        if isinstance(b, None_):
            tokens.extend(expression_to_tokens_with_pitch(a, d, in_condition, is_naive))
            tokens.append(("플러스", d, 0))
            return tokens
        
        tokens.extend(expression_to_tokens_with_pitch(a, d, in_condition, is_naive))
        b_d = d + 1 if is_add_like(b) else d
        tokens.append(("더하기", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(b, b_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Sub):
        a, b = expr.left, expr.right
        # None_인지 확인
        if isinstance(b, None_):
            tokens.extend(expression_to_tokens_with_pitch(a, d, in_condition, is_naive))
            tokens.append(("마이너스", d, 0))
            return tokens
        
        tokens.extend(expression_to_tokens_with_pitch(a, d, in_condition, is_naive))
        b_d = d + 1 if is_add_like(b) else d
        tokens.append(("빼기", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(b, b_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, PlusMinus):
        a, b = expr.left, expr.right
        tokens.extend(expression_to_tokens_with_pitch(a, d, in_condition, is_naive))
        b_d = d + 1 if is_add_like(b) else d
        tokens.append(("플마", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(b, b_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, MinusPlus):
        a, b = expr.left, expr.right
        tokens.extend(expression_to_tokens_with_pitch(a, d, in_condition, is_naive))
        b_d = d + 1 if is_add_like(b) else d
        tokens.append(("마플", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(b, b_d, in_condition, is_naive))
        return tokens

    # ==================== 2항 연산자: 곱셈/나눗셈 ====================

    if isinstance(expr, Divide):
        a, b = expr.left, expr.right
        a_d = d + 1 if is_add_like(a) else d
        # b의 경우
        if is_add_mult_chain(b):
            b_d = d + 1
        else:
            b_d = d
        tokens.extend(expression_to_tokens_with_pitch(a, a_d, in_condition, is_naive))
        tokens.append(("나누기", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(b, b_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Slash):  # a/b
        a, b = expr.left, expr.right
        a_d = d + 1 if is_operator(a) else d
        b_d = d + 1 if is_operator(b) else d
        # 읽기: b 분의 a
        tokens.extend(expression_to_tokens_with_pitch(b, b_d, in_condition, is_naive))
        tokens.append(("분의", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(a, a_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Frac):  # Frac(num, denom) -> denom/num
        num_part = expr.num
        denom_part = expr.denom
        num_d = d + 1 if is_operator(num_part) else d
        denom_d = d + 1 if is_operator(denom_part) else d
        # 읽기: denom 분의 num
        tokens.extend(expression_to_tokens_with_pitch(denom_part, denom_d, in_condition, is_naive))
        tokens.append(("분의", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(num_part, num_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, MixedFrac):  # whole num/denom
        whole = expr.whole
        num_part = expr.num
        denom_part = expr.denom
        
        whole_token = expression_to_tokens_with_pitch(whole, d, in_condition, is_naive)
        tokens.extend(whole_token)

        head = pick_head_for_particle_from_tokens(whole_token)
        nen, iga, reul, gwa, euro = get_particle(head)
        tokens.append((gwa, d, 0))
        
        denom_d = d + 1 if is_operator(denom_part) else d
        num_d = d + 1 if is_operator(num_part) else d
        
        tokens.extend(expression_to_tokens_with_pitch(denom_part, denom_d, in_condition, is_naive))
        tokens.append(("분의", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(num_part, num_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Mult):
        a, b = expr.left, expr.right
        a_d = d + 1 if is_add_like(a) else d
        b_d = d + 1 if is_add_mult_chain(b) else d
        tokens.extend(expression_to_tokens_with_pitch(a, a_d, in_condition, is_naive))
        tokens.append(("곱하기", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(b, b_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, ImplicitMult):
        a, b = expr.left, expr.right
        
        # a의 pitch 결정
        if isinstance(a, (Factorial, Prime, Power, Bar, Mult, ImplicitMult)) or (not is_operator(a)):
            a_d = d
        else:
            a_d = d + 1
        
        # b의 pitch 결정
        b_d = d + 1 if is_add_mult_chain(b) else d
        
        a_tokens = expression_to_tokens_with_pitch(a, a_d, in_condition, is_naive)
        b_tokens = expression_to_tokens_with_pitch(b, b_d, in_condition, is_naive)
        
        # 규칙 1: 둘 다 d+1이면 b의 첫 토큰 volume +1
        if a_tokens and b_tokens:
            if a_tokens[0][1] == d + 1 and b_tokens[0][1] == d + 1:
                t, p, v = b_tokens[0]
                b_tokens[0] = (t, p, v + 1)
        
        # 규칙 2: 마지막과 첫 pitch가 d+2 이상이면 "효과음" 삽입
        if a_tokens and b_tokens:
            last_p = a_tokens[-1][1]
            first_p = b_tokens[0][1]
            if last_p >= d + 2:
                tokens.extend(a_tokens)
        #        tokens.append(("곱", d, 0))
                tokens.extend(b_tokens)
                return tokens
        
        tokens.extend(a_tokens)
        tokens.extend(b_tokens)
        return tokens

    # ==================== Power ====================

    if isinstance(expr, Power):
        base = expr.base
        expo = expr.expo

        # --- [특수 처리] 삼각함수의 거듭제곱: sin^a x 형태로 읽기 ---
        if isinstance(base, (Sin, Cos, Tan, Sec, Csc, Cot)):
            # 삼각함수 이름 매핑
            if isinstance(base, Sin):
                name = "사인"
            elif isinstance(base, Cos):
                name = "코사인"
            elif isinstance(base, Tan):
                name = "탄젠트"
            elif isinstance(base, Sec):
                name = "시컨트"
            elif isinstance(base, Csc):
                name = "코시컨트"
            else:  # Cot
                name = "코탄젠트"

            # 함수 이름: 현재 depth d
            tokens.append((name, d, 0))

            # 지수 처리
            if is_integer_value(expo) and str(expo.val) == "2":
                # sin^2 x → "사인 제곱 엑스"
                tokens.append(("제곱", d, 0))
            else:
                # sin^3 x → "사인 삼 제곱 엑스"
                # sin^{10.6} x → "사인 십 점 육 제곱 엑스"
                # 지수 표현을 그대로 읽은 다음 "제곱"을 붙인다.
                tokens.extend(expression_to_tokens_with_pitch(expo, d, in_condition, is_naive))
                tokens.append(("제곱", d, 0))

            # 인자 x
            arg = base.arg
            arg_d = d + 1 if is_mid_or_postfix(arg) else d
            tokens.extend(expression_to_tokens_with_pitch(arg, arg_d, in_condition, is_naive))
            return tokens

        # --- [기본 처리] 일반적인 거듭제곱 ---
        # base pitch
        if isinstance(base, Power) or is_infix(base):
            base_d = d + 1
        else:
            base_d = d

        tokens.extend(expression_to_tokens_with_pitch(base, base_d, in_condition, is_naive))

        # 지수가 2면 그냥 "제곱"만
        if is_integer_value(expo) and str(expo.val) == "2":
            tokens.append(("제곱", d, 0))
        else:
            tokens.append(("의", d, 0))
            tokens.extend(expression_to_tokens_with_pitch(expo, d, in_condition, is_naive))
            tokens.append(("제곱", d, 0))

        return tokens


    # ==================== SQRT ====================

    if isinstance(expr, SQRT):
        radicand = expr.radicand
        index = expr.index
        
        index_d = d + 1 if is_mid_or_postfix(index) else d
        radicand_d = d + 1 if is_mid_or_postfix(radicand) else d
        
        # 제곱근인지 확인
        index_str = str(index)
        if index_str == "2":
            tokens.append(("루트", d, 0))
            tokens.extend(expression_to_tokens_with_pitch(radicand, radicand_d, in_condition, is_naive))
        else:
            tokens.extend(expression_to_tokens_with_pitch(index, index_d, in_condition, is_naive))
            tokens.append(("제곱근", d, 0))
            tokens.extend(expression_to_tokens_with_pitch(radicand, radicand_d, in_condition, is_naive))
        return tokens

# ==================== 비교/서술 연산자 ====================

    if isinstance(expr, Eq):
        a, b = expr.left, expr.right
        a_tokens = expression_to_tokens_with_pitch(a, d, in_condition, is_naive)
        tokens.extend(a_tokens)

        # a 는/은
        if a_tokens:
            head = pick_head_for_particle_from_tokens(a_tokens)
            nen, iga, reul, gwa, euro = get_particle(head)
            tokens.append((nen, d, 0))  # "은/는"
        else:
            tokens.append(("은", d, 0))

        # 오른쪽이 또 서술형이면: "a는 (right의 가장 왼쪽)이고 (right 전체)"
        if is_descriptive_operator(b):
            leftmost = get_leftmost_operand(b)
            leftmost_tokens = expression_to_tokens_with_pitch(leftmost, d, in_condition, is_naive)
            tokens.extend(leftmost_tokens)
            tokens.append(("이고", d, 0))
            tokens.extend(expression_to_tokens_with_pitch(b, d, in_condition, is_naive))
            return tokens

        # 일반 a = b : "a는 b"
        tokens.extend(expression_to_tokens_with_pitch(b, d, in_condition, is_naive))
        return tokens


    if isinstance(expr, Neq):
        a, b = expr.left, expr.right
        a_tokens = expression_to_tokens_with_pitch(a, d, in_condition, is_naive)
        tokens.extend(a_tokens)

        # a 는/은
        if a_tokens:
            head_a = pick_head_for_particle_from_tokens(a_tokens)
            nen, iga, reul, gwa, euro = get_particle(head_a)
            tokens.append((nen, d, 0))
        else:
            tokens.append(("은", d, 0))

        # 오른쪽이 서술형이면: 
        #   "a는 (right의 가장 왼쪽 피연산자)가 아니고 (right 전체)"
        if is_descriptive_operator(b):
            b_head_expr = get_leftmost_operand(b)
            b_head_tokens = expression_to_tokens_with_pitch(b_head_expr, d, in_condition, is_naive)
            tokens.extend(b_head_tokens)

            if b_head_tokens:
                head_b = pick_head_for_particle_from_tokens(b_head_tokens)
                nen2, iga2, reul2, gwa2, euro2 = get_particle(head_b)
                tokens.append((iga2, d, 0))  # "이/가"
            else:
                tokens.append(("이", d, 0))

            tokens.append(("아니고", d, 0))

            # 나머지 서술 전체 (예: 3 < 4 → "삼은 사보다 작다")
            tokens.extend(expression_to_tokens_with_pitch(b, d, in_condition, is_naive))
            return tokens

        # 일반 a ≠ b : "a는 b가 아니다"
        b_tokens = expression_to_tokens_with_pitch(b, d, in_condition, is_naive)
        tokens.extend(b_tokens)
        tokens.append(("가", d, 0))
        tokens.append(("아니다", d, 0))
        return tokens


    if isinstance(expr, Less):
        a, b = expr.left, expr.right
        a_tokens = expression_to_tokens_with_pitch(a, d, in_condition, is_naive)
        tokens.extend(a_tokens)

        # a 는/은 (비교류는 주어에 은/는 사용)
        if a_tokens:
            head = pick_head_for_particle_from_tokens(a_tokens)
            nen, iga, reul, gwa, euro = get_particle(head)
            tokens.append((nen, d, 0))
        else:
            tokens.append(("은", d, 0))

        # 오른쪽이 서술형이면: 
        #   "a는 (right의 가장 왼쪽)보다 작고 (right 전체)"
        if is_descriptive_operator(b):
            leftmost = get_leftmost_operand(b)
            leftmost_tokens = expression_to_tokens_with_pitch(leftmost, d, in_condition, is_naive)
            tokens.extend(leftmost_tokens)
            tokens.append(("보다", d, 0))
            tokens.append(("작고", d, 0))
            tokens.extend(expression_to_tokens_with_pitch(b, d, in_condition, is_naive))
            return tokens

        # 일반 a < b : "a는 b보다 작다"
        b_tokens = expression_to_tokens_with_pitch(b, d, in_condition, is_naive)
        tokens.extend(b_tokens)
        tokens.append(("보다", d, 0))
        tokens.append(("작다", d, 0))
        return tokens


    if isinstance(expr, Leq):
        a, b = expr.left, expr.right
        a_tokens = expression_to_tokens_with_pitch(a, d, in_condition, is_naive)
        tokens.extend(a_tokens)

        if a_tokens:
            head = pick_head_for_particle_from_tokens(a_tokens)
            nen, iga, reul, gwa, euro = get_particle(head)
            tokens.append((nen, d, 0))
        else:
            tokens.append(("은", d, 0))

        # a ≤ (서술형 b) : "a는 (b의 가장 왼쪽) 이하이고 (b 전체)"
        if is_descriptive_operator(b):
            leftmost = get_leftmost_operand(b)
            leftmost_tokens = expression_to_tokens_with_pitch(leftmost, d, in_condition, is_naive)
            tokens.extend(leftmost_tokens)
            tokens.append(("이하이고", d, 0))
            tokens.extend(expression_to_tokens_with_pitch(b, d, in_condition, is_naive))
            return tokens

        # 일반 a ≤ b
        tokens.extend(expression_to_tokens_with_pitch(b, d, in_condition, is_naive))
        tokens.append(("이하이다", d, 0))
        return tokens


    if isinstance(expr, Greater):
        a, b = expr.left, expr.right
        a_tokens = expression_to_tokens_with_pitch(a, d, in_condition, is_naive)
        tokens.extend(a_tokens)

        if a_tokens:
            head = pick_head_for_particle_from_tokens(a_tokens)
            nen, iga, reul, gwa, euro = get_particle(head)
            tokens.append((nen, d, 0))
        else:
            tokens.append(("은", d, 0))

        # a > (서술형 b) : "a는 (b의 가장 왼쪽)보다 크고 (b 전체)"
        if is_descriptive_operator(b):
            leftmost = get_leftmost_operand(b)
            leftmost_tokens = expression_to_tokens_with_pitch(leftmost, d, in_condition, is_naive)
            tokens.extend(leftmost_tokens)
            tokens.append(("보다", d, 0))
            tokens.append(("크고", d, 0))
            tokens.extend(expression_to_tokens_with_pitch(b, d, in_condition, is_naive))
            return tokens

        # 일반 a > b
        tokens.extend(expression_to_tokens_with_pitch(b, d, in_condition, is_naive))
        tokens.append(("보다", d, 0))
        tokens.append(("크다", d, 0))
        return tokens


    if isinstance(expr, Geq):
        a, b = expr.left, expr.right
        a_tokens = expression_to_tokens_with_pitch(a, d, in_condition, is_naive)
        tokens.extend(a_tokens)

        if a_tokens:
            head = pick_head_for_particle_from_tokens(a_tokens)
            nen, iga, reul, gwa, euro = get_particle(head)
            tokens.append((nen, d, 0))
        else:
            tokens.append(("은", d, 0))

        # a ≥ (서술형 b) : "a는 (b의 가장 왼쪽) 이상이고 (b 전체)"
        if is_descriptive_operator(b):
            leftmost = get_leftmost_operand(b)
            leftmost_tokens = expression_to_tokens_with_pitch(leftmost, d, in_condition, is_naive)
            tokens.extend(leftmost_tokens)
            tokens.append(("이상이고", d, 0))
            tokens.extend(expression_to_tokens_with_pitch(b, d, in_condition, is_naive))
            return tokens

        # 일반 a ≥ b
        tokens.extend(expression_to_tokens_with_pitch(b, d, in_condition, is_naive))
        tokens.append(("이상이다", d, 0))
        return tokens

    # ==================== 집합 관련 ====================

    if isinstance(expr, SetIn):
        elem = expr.elem
        set_expr = expr.set_expr

        # 왼쪽 원소 A
        elem_tokens = expression_to_tokens_with_pitch(elem, d, in_condition, is_naive)
        tokens.extend(elem_tokens)

        # A 은/는
        if elem_tokens:
            head = pick_head_for_particle_from_tokens(elem_tokens)
            nen, iga, reul, gwa, euro = get_particle(head)
            tokens.append((nen, d, 0))  # "은/는"
        else:
            tokens.append(("은", d, 0))

        # 오른쪽이 서술형(비교/집합관계/합동/닮음/평행/수직 등)이면 체인 처리
        if is_descriptive_operator(set_expr):
            leftmost = get_leftmost_operand(set_expr)
            leftmost_tokens = expression_to_tokens_with_pitch(leftmost, d, in_condition, is_naive)
            tokens.extend(leftmost_tokens)
            tokens.append(("의", d, 0))
            tokens.append(("원소이고", d, 0))
            tokens.extend(expression_to_tokens_with_pitch(set_expr, d, in_condition, is_naive))
            return tokens

        # 일반: "A는 B의 원소이다" 또는 조건부 맥락에서는 "A는 B의 원소"
        set_tokens = expression_to_tokens_with_pitch(set_expr, d, in_condition, is_naive)
        tokens.extend(set_tokens)
        tokens.append(("의", d, 0))
        if in_condition:
            tokens.append(("원소", d, 0))  # 조건부 맥락: "이다" 생략
        else:
            tokens.append(("원소이다", d, 0))  # 일반: "이다" 포함
        return tokens


    if isinstance(expr, SetNotIn):
        elem = expr.elem
        set_expr = expr.set_expr

        # 왼쪽 원소 A
        elem_tokens = expression_to_tokens_with_pitch(elem, d, in_condition, is_naive)
        tokens.extend(elem_tokens)

        # A 은/는
        if elem_tokens:
            head = pick_head_for_particle_from_tokens(elem_tokens)
            nen, iga, reul, gwa, euro = get_particle(head)
            tokens.append((nen, d, 0))
        else:
            tokens.append(("은", d, 0))

        # A \notin (서술형) → "A는 B의 원소가 아니고 B는 C와 같다" 같은 체인
        if is_descriptive_operator(set_expr):
            leftmost = get_leftmost_operand(set_expr)
            leftmost_tokens = expression_to_tokens_with_pitch(leftmost, d, in_condition, is_naive)
            tokens.extend(leftmost_tokens)
            tokens.append(("의", d, 0))
            tokens.append(("원소가", d, 0))
            tokens.append(("아니고", d, 0))
            tokens.extend(expression_to_tokens_with_pitch(set_expr, d, in_condition, is_naive))
            return tokens

        # 일반: "A는 B의 원소가 아니다"
        set_tokens = expression_to_tokens_with_pitch(set_expr, d, in_condition, is_naive)
        tokens.extend(set_tokens)
        tokens.append(("의", d, 0))
        tokens.append(("원소가", d, 0))
        tokens.append(("아니다", d, 0))
        return tokens


    if isinstance(expr, SetBuilder):
        var = expr.var
        cond = expr.condition
        tokens.append(("집합", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(var, d, in_condition, is_naive))
        tokens.append(("바", d, 0))
        # 조건은 살짝 강조
        tokens.extend(expression_to_tokens_with_pitch(cond, d + 1, in_condition, is_naive))
        return tokens


    if isinstance(expr, SetRoster):
        elements = expr.elements
        tokens.append(("집합", d, 0))
        for i, elem in enumerate(elements):
            elem_d = d + 1 if is_mid_or_postfix(elem) else d
            tokens.extend(expression_to_tokens_with_pitch(elem, elem_d, in_condition, is_naive))
            if i < len(elements) - 1:
                tokens.append(("콤마", d, 0))
        return tokens


    if isinstance(expr, SetNum):
        A = expr.expr
        A_d = d + 1 if is_mid_or_postfix(A) else d
        tokens.extend(expression_to_tokens_with_pitch(A, A_d, in_condition, is_naive))
        tokens.append(("의", d, 0))
        tokens.append(("원소의", d, 0))
        tokens.append(("개수", d, 0))
        return tokens


    if isinstance(expr, (SetSub, SetSup, SetNotSub, SetNotSup)):
        A, B = expr.left, expr.right

        # 왼쪽 집합 A
        A_tokens = expression_to_tokens_with_pitch(A, d, in_condition, is_naive)
        tokens.extend(A_tokens)

        # A 은/는
        if A_tokens:
            head = pick_head_for_particle_from_tokens(A_tokens)
            nen, iga, reul, gwa, euro = get_particle(head)
            tokens.append((nen, d, 0))
        else:
            tokens.append(("은", d, 0))

        # ⊂
        if isinstance(expr, SetSub):
            # A ⊂ (서술형) → "A는 B의 부분집합이고 B는 C와 같다"
            if is_descriptive_operator(B):
                leftmost = get_leftmost_operand(B)
                leftmost_tokens = expression_to_tokens_with_pitch(leftmost, d, in_condition, is_naive)
                tokens.extend(leftmost_tokens)
                tokens.append(("의", d, 0))
                tokens.append(("부분집합이고", d, 0))
                tokens.extend(expression_to_tokens_with_pitch(B, d, in_condition, is_naive))
                return tokens

            # 일반: "A는 B의 부분집합이다"
            tokens.extend(expression_to_tokens_with_pitch(B, d, in_condition, is_naive))
            tokens.append(("의", d, 0))
            tokens.append(("부분집합이다", d, 0))
            return tokens

        # ⊄
        if isinstance(expr, SetNotSub):
            tokens.extend(expression_to_tokens_with_pitch(B, d, in_condition, is_naive))
            tokens.append(("의", d, 0))
            tokens.append(("부분집합이", d, 0))
            tokens.append(("아니다", d, 0))
            return tokens

        # ⊃
        if isinstance(expr, SetSup):
            # A ⊃ (서술형) → "A는 B를 포함하고 B는 C와 같다"
            if is_descriptive_operator(B):
                leftmost = get_leftmost_operand(B)
                leftmost_tokens = expression_to_tokens_with_pitch(leftmost, d, in_condition, is_naive)
                tokens.extend(leftmost_tokens)
                tokens.append(("을", d, 0))
                tokens.append(("포함하고", d, 0))
                tokens.extend(expression_to_tokens_with_pitch(B, d, in_condition, is_naive))
                return tokens

            # 일반: "A는 B를 포함한다"
            tokens.extend(expression_to_tokens_with_pitch(B, d, in_condition, is_naive))
            tokens.append(("을", d, 0))
            tokens.append(("포함한다", d, 0))
            return tokens

        # ⊅
        # SetNotSup
        tokens.extend(expression_to_tokens_with_pitch(B, d, in_condition, is_naive))
        tokens.append(("을", d, 0))
        tokens.append(("포함하지", d, 0))
        tokens.append(("않는다", d, 0))
        return tokens

    if isinstance(expr, SetCup):
        A, B = expr.left, expr.right
        # 합집합이 교집합을 포함하는 경우 pitch up (우선순위 차이)
        A_d = d + 1 if isinstance(A, SetCap) else d
        B_d = d + 1 if is_mid_or_postfix(B) else d
        tokens.extend(expression_to_tokens_with_pitch(A, A_d, in_condition, is_naive))
        tokens.append(("합집합", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(B, B_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, SetCap):
        A, B = expr.left, expr.right
        # 교집합이 합집합을 포함하는 경우 pitch up (우선순위 차이)
        A_d = d + 1 if isinstance(A, SetCup) else d
        B_d = d + 1 if is_mid_or_postfix(B) else d
        tokens.extend(expression_to_tokens_with_pitch(A, A_d, in_condition, is_naive))
        tokens.append(("교집합", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(B, B_d, in_condition, is_naive))
        return tokens

    # ==================== 첨자 ====================

    if isinstance(expr, Subscript):
        base = expr.base
        sub = expr.sub
        
        base_d = d + 1 if is_operator(base) else d
        base_tokens = expression_to_tokens_with_pitch(base, base_d, in_condition, is_naive)
        
        # sub이 정수인지 확인
        is_int = is_integer_value(sub)
        sub_d = d if is_int else d + 1
        sub_tokens = expression_to_tokens_with_pitch(sub, sub_d, in_condition, is_naive)
        
        # 비정수면 첫 토큰 volume -1
        if not is_int and sub_tokens:
            t, p, v = sub_tokens[0]
            sub_tokens[0] = (t, p, v - 1)
        
        tokens.extend(base_tokens)
        tokens.extend(sub_tokens)
        return tokens

    if isinstance(expr, Superscript):
        base = expr.base
        sup = expr.sup
        
        base_d = d + 1 if is_operator(base) else d
        base_tokens = expression_to_tokens_with_pitch(base, base_d, in_condition, is_naive)
        
        is_int = is_integer_value(sup)
        sup_d = d if is_int else d + 1
        sup_tokens = expression_to_tokens_with_pitch(sup, sup_d, in_condition, is_naive)
        
        tokens.extend(base_tokens)
        
        # 비정수면 효과음 추가
        if not is_int:
            tokens.append(("[beep_up]", d, 0))
            tokens.extend(sup_tokens)
            tokens.append(("[beep_down]", d, 0))
        else:
            tokens.extend(sup_tokens)
        
        return tokens

    # ==================== 논리/명제 ====================

    if isinstance(expr, (Rimpl, Limpl, Rsufficient, Lsufficient, Biconditional, Iff)):
        p, q = expr.left, expr.right
        if isinstance(expr, (Biconditional, Iff)):
            p_d = d + 1 if is_operator(p) else d
            q_d = d + 1 if is_operator(q) else d
        else:
            p_d = d + 1 if is_mid_or_postfix(p) else d
            q_d = d + 1 if is_mid_or_postfix(q) else d

        # 왼쪽 명제 p 먼저 읽기
        p_tokens = expression_to_tokens_with_pitch(p, p_d, in_condition, is_naive)
        if isinstance(expr, Iff):
            tokens.append(("참으로", d, 0))
        tokens.extend(p_tokens)

        # 왼쪽 명제에 붙일 조사 계산용
        if p_tokens:
            head = pick_head_for_particle_from_tokens(p_tokens)
            nen, iga, reul, gwa, euro = get_particle(head)
        else:
            # p가 비어 있는 극단적인 경우 기본값
            nen, iga, reul, gwa, euro = ("는", "이", "을", "과", "으로")

        # 오른쪽 명제 토큰
        q_tokens = expression_to_tokens_with_pitch(q, q_d, in_condition, is_naive)

        if isinstance(expr, Rimpl):
            # P이면 Q
            tokens.append(("이면", d, 0))
            tokens.extend(q_tokens)
            return tokens

        elif isinstance(expr, Limpl):
            # P는 Q에 의해 함의된다
            tokens.append((nen, d, 0))         # "는/은"
            tokens.extend(q_tokens)            # Q
            tokens.append(("에", d, 0))
            tokens.append(("의해", d, 0))
            tokens.append(("함의된다", d, 0))
            return tokens

        elif isinstance(expr, Rsufficient):
            # P는 Q의 충분조건이다
            tokens.append((nen, d, 0))         # "는/은"
            tokens.extend(q_tokens)            # Q
            tokens.append(("의", d, 0))
            tokens.append(("충분조건이다", d, 0))
            return tokens

        elif isinstance(expr, Biconditional):
            # P는 Q의 필요충분조건이다
            tokens.append((nen, d, 0))
            tokens.extend(q_tokens)
            tokens.append(("의", d, 0))
            tokens.append(("필요충분조건이다", d, 0))
            return tokens

        elif isinstance(expr, Iff):
            # 참으로 P는 Q의 필요충분조건이다
            tokens.append((nen, d, 0))
            tokens.extend(q_tokens)
            tokens.append(("의", d, 0))
            tokens.append(("필요충분조건이다", d, 0))
            return tokens

        else:  # Lsufficient
            # P는 Q의 필요조건이다
            tokens.append((nen, d, 0))         # "는/은"
            tokens.extend(q_tokens)            # Q
            tokens.append(("의", d, 0))
            tokens.append(("필요조건이다", d, 0))
            return tokens


    if isinstance(expr, Prop):
        symbol = expr.symbol
        statement = expr.statement
        tokens.append(("명제", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(symbol, d, in_condition, is_naive))
        # tokens.append(("콜론", d, 0))        
        tokens.extend(expression_to_tokens_with_pitch(statement, d, in_condition, is_naive))
        return tokens

    if isinstance(expr, FuncDef):
        func = expr.func
        domain = expr.domain
        codomain = expr.codomain

        tokens.append(("함수", d, 0))
        func_tokens = expression_to_tokens_with_pitch(func, d, in_condition, is_naive)
        tokens.extend(func_tokens)

        if func_tokens:
            func_head = func_tokens[-1][0]
            nen = get_particle(func_head)[0]
        else:
            nen = "는"
        tokens.append((nen, d, 0))

        domain_tokens = expression_to_tokens_with_pitch(domain, d, in_condition, is_naive)
        tokens.extend(domain_tokens)

        tokens.append(("에서", d, 0))

        # 공역 Y
        codomain_tokens = expression_to_tokens_with_pitch(codomain, d, in_condition, is_naive)
        tokens.extend(codomain_tokens)

        # codomain 뒤 "로/으로"
        if codomain_tokens:
            codomain_head = codomain_tokens[-1][0]
            euro2 = get_particle(codomain_head)[4]
        else:
            euro2 = "으로"
        tokens.append((euro2, d, 0))

        tokens.append(("가는", d, 0))
        tokens.append(("함수", d, 0))
        return tokens

    # ==================== 함수 ====================

    if isinstance(expr, Func):
        name = expr.name
        args = expr.args
        
        name_d = d + 1 if is_mid_or_postfix(name) else d
        tokens.extend(expression_to_tokens_with_pitch(name, name_d, in_condition, is_naive))
        
        for i, arg in enumerate(args):
            arg_d = d + 1 if is_mid_or_postfix(arg) else d
            tokens.extend(expression_to_tokens_with_pitch(arg, arg_d, in_condition, is_naive))
            if i < len(args) - 1:
                tokens.append(("콤마", d, 0))
        
        return tokens

    if isinstance(expr, FuncInv):
        name = expr.name
        args = expr.args
        
        name_d = d + 1 if is_operator(name) else d
        tokens.extend(expression_to_tokens_with_pitch(name, name_d, in_condition, is_naive))
        tokens.append(("인버스", d, 0))
        
        for i, arg in enumerate(args):
            arg_d = d + 1 if is_mid_or_postfix(arg) else d
            tokens.extend(expression_to_tokens_with_pitch(arg, arg_d, in_condition, is_naive))
            if i < len(args) - 1:
                tokens.append(("콤마", d, 0))
        
        return tokens

    # ==================== 조합론 ====================

    if isinstance(expr, Perm):
        n, r = expr.n, expr.r
        n_d = d + 1 if is_mid_or_postfix(n) else d
        r_d = d + 1 if is_mid_or_postfix(r) else d
        tokens.append(("순열", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(n, n_d, in_condition, is_naive))
        tokens.append(("피", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(r, r_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Comb):
        n, r = expr.n, expr.r
        n_d = d + 1 if is_mid_or_postfix(n) else d
        r_d = d + 1 if is_mid_or_postfix(r) else d
        tokens.append(("조합", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(n, n_d, in_condition, is_naive))
        tokens.append(("씨", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(r, r_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, RepeatedPermu):
        n, r = expr.n, expr.r
        n_d = d + 1 if is_mid_or_postfix(n) else d
        r_d = d + 1 if is_mid_or_postfix(r) else d
        tokens.append(("중복순열", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(n, n_d, in_condition, is_naive))
        tokens.append(("파이", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(r, r_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, RepeatedComb):
        n, r = expr.n, expr.r
        n_d = d + 1 if is_mid_or_postfix(n) else d
        r_d = d + 1 if is_mid_or_postfix(r) else d
        tokens.append(("중복조합", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(n, n_d, in_condition, is_naive))
        tokens.append(("에이치", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(r, r_d, in_condition, is_naive))
        return tokens

    # ==================== 로그/삼각함수 ====================

    if isinstance(expr, Log):
        base = expr.base
        arg = expr.arg

        arg_d = d + 1 if is_mid_or_postfix(arg) else d

        tokens.append(("로그", d, 0))

        if not isinstance(base, None_):
            tokens.extend(expression_to_tokens_with_pitch(base, d, in_condition, is_naive))
            tokens.append(("의", d, 0))

        tokens.extend(expression_to_tokens_with_pitch(arg, arg_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Ln):
        arg = expr.arg
        arg_d = d + 1 if is_mid_or_postfix(arg) else d
        tokens.append(("엘엔", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(arg, arg_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Sin):
        arg = expr.arg
        arg_d = d + 1 if is_mid_or_postfix(arg) else d
        tokens.append(("사인", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(arg, arg_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Cos):
        arg = expr.arg
        arg_d = d + 1 if is_mid_or_postfix(arg) else d
        tokens.append(("코사인", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(arg, arg_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Tan):
        arg = expr.arg
        arg_d = d + 1 if is_mid_or_postfix(arg) else d
        tokens.append(("탄젠트", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(arg, arg_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Sec):
        arg = expr.arg
        arg_d = d + 1 if is_mid_or_postfix(arg) else d
        tokens.append(("시컨트", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(arg, arg_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Csc):
        arg = expr.arg
        arg_d = d + 1 if is_mid_or_postfix(arg) else d
        tokens.append(("코시컨트", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(arg, arg_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Cot):
        arg = expr.arg
        arg_d = d + 1 if is_mid_or_postfix(arg) else d
        tokens.append(("코탄젠트", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(arg, arg_d, in_condition, is_naive))
        return tokens

    # ==================== 수열/극한/급수/미적분 ====================

    if isinstance(expr, Seq):
        term = expr.term
        if not is_naive:
            tokens.append(("수열", d, 0))
        # term이 Subscript이든, Value이든, 다른 거든 전부 공평하게 재귀 처리
        tokens.extend(expression_to_tokens_with_pitch(term, d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Lim):
        var = expr.var
        to = expr.to
        expr_inner = expr.expr

        tokens.append(("리미트", d, 0))

        # 변수 부분: n이, x가 ...
        var_tokens = expression_to_tokens_with_pitch(var, d, in_condition, is_naive)
        tokens.extend(var_tokens)
        if var_tokens:
            var_head = var_tokens[-1][0]
            _, iga, _, _, _ = get_particle(var_head)
        else:
            iga = "이"
        tokens.append((iga, d, 0))   # 이/가

        # → 값 부분: 무한대로, 0으로 ...
        to_tokens = expression_to_tokens_with_pitch(to, d, in_condition, is_naive)
        tokens.extend(to_tokens)
        if to_tokens:
            to_head = to_tokens[-1][0]
            _, _, _, _, euro = get_particle(to_head)
        else:
            euro = "으로"
        tokens.append((euro, d, 0))  # 로/으로

        tokens.append(("갈", d, 0))
        tokens.append(("때", d, 0))

        # 본문은 기존처럼 pitch 조정
        expr_d = d + 1 if is_mid_or_postfix(expr_inner) else d
        tokens.extend(expression_to_tokens_with_pitch(expr_inner, expr_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Sum):
        term = expr.term
        var = expr.var
        start = expr.start
        end = expr.end

        tokens.append(("시그마", d, 0))

        # 시그마 아래 변수: n은, k는 ...
        var_tokens = expression_to_tokens_with_pitch(var, d, in_condition, is_naive)
        tokens.extend(var_tokens)
        if var_tokens:
            var_head = var_tokens[-1][0]
            nen, _, _, _, _ = get_particle(var_head)
        else:
            nen = "는"
        tokens.append((nen, d, 0))   # 은/는

        # 범위: a부터 b까지
        tokens.extend(expression_to_tokens_with_pitch(start, d, in_condition, is_naive))
        tokens.append(("부터", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(end, d, in_condition, is_naive))
        tokens.append(("까지", d, 0))

        # 합의 항
        term_d = d + 1 if is_mid_or_postfix(term) else d
        tokens.extend(expression_to_tokens_with_pitch(term, term_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Diff):
        y = expr.y
        x = expr.x
        n = expr.n

        n_str = str(n)

        # 앞쪽 "디 ..."
        tokens.append(("디", d, 0))

        # n=1,2,그 외 케이스 분기
        if n_str == "1":
            # 아무 것도 안 붙임
            pass
        elif n_str == "2":
            # "이 제곱"이 아니라 그냥 "제곱"
            tokens.append(("제곱", d, 0))
        else:
            # 일반적인 n제곱
            tokens.extend(expression_to_tokens_with_pitch(n, d, in_condition, is_naive))
            tokens.append(("제곱", d, 0))

        # y 부분
        y_d = d + 1 if is_mid_or_postfix(y) else d
        tokens.extend(expression_to_tokens_with_pitch(y, y_d, in_condition, is_naive))

        # 중간 "디"
        tokens.append(("디", d, 0))

        # x 부분
        x_d = d + 1 if is_mid_or_postfix(x) else d
        tokens.extend(expression_to_tokens_with_pitch(x, x_d, in_condition, is_naive))

        # 뒤쪽 지수
        if n_str == "1":
            pass
        elif n_str == "2":
            tokens.append(("제곱", d, 0))
        else:
            tokens.extend(expression_to_tokens_with_pitch(n, d, in_condition, is_naive))
            tokens.append(("제곱", d, 0))

        return tokens


    if isinstance(expr, Integral):
        lower = expr.lower
        upper = expr.upper
        integrand = expr.integrand
        var = expr.var
        
        tokens.append(("인티그럴", d, 0))
        
        # 하한/상한 처리
        if not isinstance(lower, None_):
            tokens.extend(expression_to_tokens_with_pitch(lower, d, in_condition, is_naive))
            tokens.append(("부터", d, 0))
        
        if not isinstance(upper, None_):
            tokens.extend(expression_to_tokens_with_pitch(upper, d, in_condition, is_naive))
            tokens.append(("까지", d, 0))
        
        integrand_d = d + 1 if is_mid_or_postfix(integrand) else d
        tokens.extend(expression_to_tokens_with_pitch(integrand, integrand_d, in_condition, is_naive))
        
        tokens.append(("디", d, 0))
        var_d = d + 1 if is_mid_or_postfix(var) else d
        tokens.extend(expression_to_tokens_with_pitch(var, var_d, in_condition, is_naive))
        
        return tokens

    if isinstance(expr, Integrated):
        antiderivative = expr.antiderivative
        lower = expr.lower
        upper = expr.upper
        
        tokens.extend(expression_to_tokens_with_pitch(antiderivative, d + 1, in_condition, is_naive))
        tokens.extend(expression_to_tokens_with_pitch(lower, d, in_condition, is_naive))
        tokens.append(("부터", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(upper, d, in_condition, is_naive))
        tokens.append(("까지", d, 0))
        return tokens

    # ================== 구간 ==================
    if isinstance(expr, ClosedInterval):
        tokens.extend(expression_to_tokens_with_pitch(expr.left, d, in_condition, is_naive))
        tokens.append(("초과", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(expr.right, d, in_condition, is_naive))
        tokens.append(("미만", d, 0))        
        return tokens

    if isinstance(expr, ClosedOpenInterval):
        tokens.extend(expression_to_tokens_with_pitch(expr.left, d, in_condition, is_naive))
        tokens.append(("이상", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(expr.right, d, in_condition, is_naive))
        tokens.append(("이하", d, 0))        
        return tokens

    if isinstance(expr, OpenClosedInterval):
        tokens.extend(expression_to_tokens_with_pitch(expr.left, d, in_condition, is_naive))
        tokens.append(("초과", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(expr.right, d, in_condition, is_naive))
        tokens.append(("이하", d, 0))        
        return tokens

    if isinstance(expr, OpenInterval):
        tokens.extend(expression_to_tokens_with_pitch(expr.left, d, in_condition, is_naive))
        tokens.append(("초과", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(expr.right, d, in_condition, is_naive))
        tokens.append(("미만", d, 0))        
        return tokens


    # ==================== 확률 ====================

    if isinstance(expr, Prob):
        event = expr.event
        condition = expr.condition
        
        if not is_naive:
            tokens.append(("확률", d, 0))
        tokens.append(("피", d, 0))
        
        # event의 pitch level 결정: 연산자/서술자면 d+1, 단순 원소면 d
        if is_operator(event) or is_mid_or_postfix(event):
            event_d = d + 1
        else:
            event_d = d
        tokens.extend(expression_to_tokens_with_pitch(event, event_d, in_condition=True, is_naive=is_naive))
        
        if condition is not None and not isinstance(condition, None_):
            tokens.append(("바", d, 0))
            # condition의 pitch level 결정: 연산자/서술자면 d+1, 단순 원소면 d
            if is_operator(condition) or is_mid_or_postfix(condition):
                condition_d = d + 1
            else:
                condition_d = d
            tokens.extend(expression_to_tokens_with_pitch(condition, condition_d, in_condition=True, is_naive=is_naive))
        
        return tokens

    # ==================== 기하 ====================

    if isinstance(expr, Triangle):
        vertices = expr.vertices
        tokens.append(("삼각형", d, 0))
        # vertices는 문자열이므로 Text로 변환
        tokens.extend(expression_to_tokens_with_pitch(Text(vertices), d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Angle):
        vertices = expr.vertices
        tokens.append(("각", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(Text(vertices), d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Arc):
        vertices = expr.vertices
        tokens.append(("호", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(Text(vertices), d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Point):
        name = expr.name
        args = expr.args
        
        if name is not None and not isinstance(name, None_):
            if not is_naive:
                tokens.append(("점", d, 0))
            tokens.extend(expression_to_tokens_with_pitch(name, d, in_condition, is_naive))
        
        for i, arg in enumerate(args):
            arg_d = d + 1 if is_operator(arg) else d
            tokens.extend(expression_to_tokens_with_pitch(arg, arg_d, in_condition, is_naive))
            if i < len(args) - 1:
                tokens.append(("콤마", d, 0))
        
        return tokens

    if isinstance(expr, Segment):
        start = expr.start
        end = expr.end
        tokens.append(("선분", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(start, d, in_condition, is_naive))
        tokens.extend(expression_to_tokens_with_pitch(end, d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Ray):
        start = expr.start
        through = expr.through
        tokens.append(("반직선", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(start, d, in_condition, is_naive))
        tokens.extend(expression_to_tokens_with_pitch(through, d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Line):
        point1 = expr.point1
        point2 = expr.point2
        tokens.append(("직선", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(point1, d, in_condition, is_naive))
        tokens.extend(expression_to_tokens_with_pitch(point2, d, in_condition, is_naive))
        return tokens

    if isinstance(expr, LineExpr):
        line = expr.line
        eq = expr.eq
        tokens.append(("직선", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(line, d, in_condition, is_naive))
        # tokens.append(("콜론", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(eq, d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Perp):
        line1 = expr.line1
        line2 = expr.line2

        # 첫 번째 직선
        line1_tokens = expression_to_tokens_with_pitch(line1, d, in_condition, is_naive)
        tokens.extend(line1_tokens)

        # "와/과" (line1 기준)
        if line1_tokens:
            last1 = line1_tokens[-1][0]
            _, _, _, wa_gwa, _ = get_particle(last1)
        else:
            wa_gwa = "와"
        tokens.append((wa_gwa, d, 0))

        if is_descriptive_operator(line2):
            # 오른쪽이 또 서술형인 경우: "수직이고 ..."
            # line2 안에서 제일 왼쪽 객체만 뽑아서 그 뒤에 "은/는"
            left2 = get_leftmost_operand(line2)
            left2_tokens = expression_to_tokens_with_pitch(left2, d, in_condition, is_naive)
            tokens.extend(left2_tokens)

            if left2_tokens:
                last2 = left2_tokens[-1][0]
                eun_neun, _, _, _, _ = get_particle(last2)
            else:
                eun_neun = "은"
            tokens.append((eun_neun, d, 0))

            # "수직이고"
            tokens.append(("수직이고", d, 0))

            # 나머지 관계는 그대로 재귀 호출
            tokens.extend(expression_to_tokens_with_pitch(line2, d, in_condition, is_naive))
        else:
            # 단일 관계: "수직이다"
            line2_tokens = expression_to_tokens_with_pitch(line2, d, in_condition, is_naive)
            tokens.extend(line2_tokens)

            if line2_tokens:
                last2 = line2_tokens[-1][0]
                eun_neun, _, _, _, _ = get_particle(last2)
            else:
                eun_neun = "은"
            tokens.append((eun_neun, d, 0))
            tokens.append(("수직이다", d, 0))

        return tokens

    if isinstance(expr, Paral):
        line1 = expr.line1
        line2 = expr.line2

        line1_tokens = expression_to_tokens_with_pitch(line1, d, in_condition, is_naive)
        tokens.extend(line1_tokens)

        if line1_tokens:
            last1 = line1_tokens[-1][0]
            _, _, _, wa_gwa, _ = get_particle(last1)
        else:
            wa_gwa = "와"
        tokens.append((wa_gwa, d, 0))

        if is_descriptive_operator(line2):
            left2 = get_leftmost_operand(line2)
            left2_tokens = expression_to_tokens_with_pitch(left2, d, in_condition, is_naive)
            tokens.extend(left2_tokens)

            if left2_tokens:
                last2 = left2_tokens[-1][0]
                eun_neun, _, _, _, _ = get_particle(last2)
            else:
                eun_neun = "은"
            tokens.append((eun_neun, d, 0))
            tokens.append(("평행이고", d, 0))
            tokens.extend(expression_to_tokens_with_pitch(line2, d, in_condition, is_naive))
        else:
            line2_tokens = expression_to_tokens_with_pitch(line2, d, in_condition, is_naive)
            tokens.extend(line2_tokens)

            if line2_tokens:
                last2 = line2_tokens[-1][0]
                eun_neun, _, _, _, _ = get_particle(last2)
            else:
                eun_neun = "은"
            tokens.append((eun_neun, d, 0))
            tokens.append(("평행이다", d, 0))

        return tokens

    if isinstance(expr, Ratio):
        for i, r in enumerate(expr.ratios):
            if i > 0:
                tokens.append(("대", d, 0))
            tokens.extend(expression_to_tokens_with_pitch(r, d, in_condition, is_naive))
        return tokens

    if isinstance(expr, InnerProduct):
        vec1 = expr.vec1
        vec2 = expr.vec2

        # a쪽 pitch: Add/Sub/±/∓ 이면 d+1, 아니면 d
        vec1_d = d + 1 if is_add_like(vec1) else d

        # b쪽 pitch: Add/Sub/±/∓/÷/×/ImplicitMult/InnerProduct 이면 d+1, 아니면 d
        if isinstance(vec2, (Add, Sub, PlusMinus, MinusPlus, Divide, Mult, ImplicitMult, InnerProduct)):
            vec2_d = d + 1
        else:
            vec2_d = d

        # 실제 토크나이즈: a 내적 b
        tokens.extend(expression_to_tokens_with_pitch(vec1, vec1_d, in_condition, is_naive))
        tokens.append(("내적", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(vec2, vec2_d, in_condition, is_naive))
        return tokens

    if isinstance(expr, Congru):
        shape1 = expr.shape1
        shape2 = expr.shape2

        shape1_tokens = expression_to_tokens_with_pitch(shape1, d, in_condition, is_naive)
        tokens.extend(shape1_tokens)

        if shape1_tokens:
            last1 = shape1_tokens[-1][0]
            _, _, _, wa_gwa, _ = get_particle(last1)
        else:
            wa_gwa = "와"
        tokens.append((wa_gwa, d, 0))

        if is_descriptive_operator(shape2):
            left2 = get_leftmost_operand(shape2)
            left2_tokens = expression_to_tokens_with_pitch(left2, d, in_condition, is_naive)
            tokens.extend(left2_tokens)

            if left2_tokens:
                last2 = left2_tokens[-1][0]
                eun_neun, _, _, _, _ = get_particle(last2)
            else:
                eun_neun = "은"
            tokens.append((eun_neun, d, 0))
            tokens.append(("합동이고", d, 0))
            tokens.extend(expression_to_tokens_with_pitch(shape2, d, in_condition, is_naive))
        else:
            shape2_tokens = expression_to_tokens_with_pitch(shape2, d, in_condition, is_naive)
            tokens.extend(shape2_tokens)

            if shape2_tokens:
                last2 = shape2_tokens[-1][0]
                eun_neun, _, _, _, _ = get_particle(last2)
            else:
                eun_neun = "은"
            tokens.append((eun_neun, d, 0))
            tokens.append(("합동이다", d, 0))

        return tokens

    if isinstance(expr, Sim):
        shape1 = expr.shape1
        shape2 = expr.shape2

        shape1_tokens = expression_to_tokens_with_pitch(shape1, d, in_condition, is_naive)
        tokens.extend(shape1_tokens)

        if shape1_tokens:
            last1 = shape1_tokens[-1][0]
            _, _, _, wa_gwa, _ = get_particle(last1)
        else:
            wa_gwa = "와"
        tokens.append((wa_gwa, d, 0))

        if is_descriptive_operator(shape2):
            left2 = get_leftmost_operand(shape2)
            left2_tokens = expression_to_tokens_with_pitch(left2, d, in_condition, is_naive)
            tokens.extend(left2_tokens)

            if left2_tokens:
                last2 = left2_tokens[-1][0]
                eun_neun, _, _, _, _ = get_particle(last2)
            else:
                eun_neun = "은"
            tokens.append((eun_neun, d, 0))
            tokens.append(("닮음이고", d, 0))
            tokens.extend(expression_to_tokens_with_pitch(shape2, d, in_condition, is_naive))
        else:
            shape2_tokens = expression_to_tokens_with_pitch(shape2, d, in_condition, is_naive)
            tokens.extend(shape2_tokens)

            if shape2_tokens:
                last2 = shape2_tokens[-1][0]
                eun_neun, _, _, _, _ = get_particle(last2)
            else:
                eun_neun = "은"
            tokens.append((eun_neun, d, 0))
            tokens.append(("닮음이다", d, 0))

        return tokens

    # ==================== 단위/기타 ====================

    if isinstance(expr, Unit):
        # Unit_dictionary에서 발음 찾기
        try:
            import csv
            import os
            dict_path = os.path.join(os.path.dirname(__file__), 'Unit_dictionary.csv')
            with open(dict_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['표기법'] == expr.unit:
                        tokens.append((row['한국어 발음'], d, 0))
                        return tokens
        except Exception as e:
            pass  # 파일이 없거나 오류가 발생하면 기본값 사용
        tokens.append((expr.unit, d, 0))  # 기본값
        return tokens

    if isinstance(expr, UnitDiv):
        num_unit = expr.num_unit
        denom_unit = expr.denom_unit
        tokens.extend(expression_to_tokens_with_pitch(num_unit, d, in_condition, is_naive))
        tokens.append(("퍼", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(denom_unit, d, in_condition, is_naive))
        return tokens

    if isinstance(expr, UnitMult):
        left = expr.left
        right = expr.right
        tokens.extend(expression_to_tokens_with_pitch(left, d, in_condition, is_naive))
        tokens.extend(expression_to_tokens_with_pitch(right, d, in_condition, is_naive))
        return tokens

    if isinstance(expr, About):
        inner = expr.expr
        inner_d = d + 1 if is_mid_or_postfix(inner) else d
        tokens.append(("약", d, 0))
        tokens.extend(expression_to_tokens_with_pitch(inner, inner_d, in_condition, is_naive))
        return tokens

    # ==================== Cases (연립방정식) ====================

    if isinstance(expr, Cases):
        for i, (case_expr, cond) in enumerate(expr.cases):
            # 1) condition을 현재 깊이 d에서 토큰화
            cond_tokens = expression_to_tokens_with_pitch(cond, d, in_condition, is_naive)

            # ----- (A) 비교 연산자: <, ≤, >, ≥ -----
            if isinstance(cond, (Less, Leq, Greater, Geq)):
                # (1) 주어 조사: 첫 "은/는" → "이/가" 로 교정
                for idx in range(1, len(cond_tokens)):
                    tok, p, v = cond_tokens[idx]
                    if tok in ("은", "는"):
                        prev_word = cond_tokens[idx - 1][0]
                        particles = get_particle(prev_word)
                        cond_tokens[idx] = (particles[1], p, v)  # "이/가"
                        break

                # (2) 문장 끝 어미: "작다" → "작을", "크다" → "클",
                #                    "이상이다" → "이상일", "이하이다" → "이하일"
                if cond_tokens:
                    last_tok, lp, lv = cond_tokens[-1]

                    if last_tok == "작다":
                        cond_tokens[-1] = ("작을", lp, lv)
                    elif last_tok == "크다":
                        cond_tokens[-1] = ("클", lp, lv)
                    elif last_tok.endswith("이상이다"):
                        cond_tokens[-1] = (last_tok[:-3] + "일", lp, lv)  # "이상 + 일"
                    elif last_tok.endswith("이하이다"):
                        cond_tokens[-1] = (last_tok[:-3] + "일", lp, lv)  # "이하 + 일"

                # (3) "… 때"만 붙인다 (중간에 따로 '일' 안 넣음)
                tokens.extend(cond_tokens)
                tokens.append(("때", d, 0))

            # ----- (B) 그 밖의 서술형 연산자들 (Eq, SetIn, Congru, Sim, …) -----
            elif is_descriptive_operator(cond):
                if cond_tokens:
                    last_tok, lp, lv = cond_tokens[-1]

                    # 끝이 "이다"로 끝나면 → "일" + "때"
                    if last_tok.endswith("이다"):
                        stem = last_tok[:-2]  # "합동이다" → "합동"
                        cond_tokens[-1] = (stem + "일", lp, lv)
                        tokens.extend(cond_tokens)
                        tokens.append(("때", d, 0))

                    # 끝이 "아니다"로 끝나면 → "아닐" + "때"
                    elif last_tok.endswith("아니다"):
                        stem = last_tok[:-3]  # "아니다" → ""
                        cond_tokens[-1] = (stem + "아닐", lp, lv)
                        tokens.extend(cond_tokens)
                        tokens.append(("때", d, 0))

                    # 그 밖의 경우 (Eq처럼 끝에 명사로 끝나는 경우): "일 때"
                    else:
                        tokens.extend(cond_tokens)
                        tokens.append(("일", d, 0))
                        tokens.append(("때", d, 0))
                else:
                    # cond가 비어 있을 일은 거의 없지만 방어적으로
                    tokens.append(("일", d, 0))
                    tokens.append(("때", d, 0))

            # ----- (C) 서술형이 아닌 일반 조건 (거의 없겠지만) -----
            else:
                tokens.extend(cond_tokens)
                tokens.append(("일", d, 0))
                tokens.append(("때", d, 0))

            # 2) 해당 case의 값
            tokens.extend(expression_to_tokens_with_pitch(case_expr, d, in_condition, is_naive))

            # 3) 마지막 case가 아니면 "그리고" 추가
            if i < len(expr.cases) - 1:
                tokens.append(("그리고", d, 0))

        return tokens

    # ==================== InlinePhrase ====================

    if isinstance(expr, InlinePhrase):
        for part in expr.parts:
            tokens.extend(expression_to_tokens_with_pitch(part, d, in_condition, is_naive))
        return tokens

    # ==================== Fallback ====================

    # 정의되지 않은 경우
    return [(str(expr), d, 0)]

grouping_pitch_test_cases=[
  {
    "id": 1,
    "expression_object": Frac(Sub(Power(Value('x'), Value(2)), Value(4)), Add(Mult(Value(2), Value('x')), Value(1))),
    "expression": "(x²-4)/(2×x+1)",
    "text_ko": "이 곱하기 엑스 더하기 일 분의 엑스 제곱 빼기 사",
    "tokens_with_depth": [
      { "token": "이",       "pitch_level": 1, "volume_level": 0 },  # 2 (denominator Mult, d=1)
      { "token": "곱하기", "pitch_level": 1, "volume_level": 0 },  # Mult(d=1)
      { "token": "엑스",   "pitch_level": 1, "volume_level": 0 },  # x in Mult
      { "token": "더하기", "pitch_level": 1, "volume_level": 0 },  # Add(d=1)
      { "token": "일",     "pitch_level": 1, "volume_level": 0 },  # 1 in Add
      { "token": "분의",   "pitch_level": 0, "volume_level": 0 },  # Slash root(d=0)
      { "token": "엑스",   "pitch_level": 1, "volume_level": 0 },  # x in Power
      { "token": "제곱",   "pitch_level": 1, "volume_level": 0 },  # Power(d=1)
      { "token": "빼기",   "pitch_level": 1, "volume_level": 0 },  # Sub(d=1)
      { "token": "사",     "pitch_level": 1, "volume_level": 0 }   # 4 in Sub
    ]
  },

  {
    "id": 2,
    "expression_object": Frac(Add(Minus(Value('b')), SQRT(Sub(Power(Value('b'), Value(2)), ImplicitMult(ImplicitMult(Value(4), Value('a')), Value('c'))))), ImplicitMult(Value(2), Value('a'))),
    "expression": "(-b+√(b²-4ac))/(2a)",
    "text_ko": "이 에이 분의 마이너스 비 더하기 루트 비 제곱 빼기 사 에이 씨",
    "tokens_with_depth": [
      { "token": "이",       "pitch_level": 1, "volume_level": 0 },  # 2 in denom Mult(d=1)
      { "token": "에이",   "pitch_level": 1, "volume_level": 0 },  # a in denom
      { "token": "분의",   "pitch_level": 0, "volume_level": 0 },  # Slash root

      { "token": "마이너스", "pitch_level": 1, "volume_level": 0 }, # Minus(d=1)
      { "token": "비",       "pitch_level": 1, "volume_level": 0 }, # b in Minus
      { "token": "더하기",   "pitch_level": 1, "volume_level": 0 }, # Add(d=1)
      { "token": "루트",     "pitch_level": 1, "volume_level": 0 }, # SQRT(d=1)

      { "token": "비",       "pitch_level": 2, "volume_level": 0 }, # b in Power(d=2)
      { "token": "제곱",     "pitch_level": 2, "volume_level": 0 }, # Power(d=2)
      { "token": "빼기",     "pitch_level": 2, "volume_level": 0 }, # Sub(d=2)

      { "token": "사",   "pitch_level": 2, "volume_level": 0 },  # 4  (inner ImplicitMult, d=3)
      { "token": "에이", "pitch_level": 2, "volume_level": 0 },  # a  (inner ImplicitMult, d=3)
      { "token": "씨",   "pitch_level": 2, "volume_level": 0 }   # c  (outer ImplicitMult의 b, d=2)
    ]
  },

  {
    "id": 3,
    "expression_object": ImplicitMult(Divide(Value(1), ImplicitMult(Add(Value(2), Value(3)), Add(Value(4), Value(5)))), Divide(ImplicitMult(Add(Value(6), Value(7)), Add(Value(8), Value(9))), Value(2))),
    "expression": "(1÷(2+3)(4+5))((6+7)(8+9)÷2)",
    "text_ko": "일 나누기 이 더하기 삼 사 더하기 오 육 더하기 칠 팔 더하기 구 나누기 이",
    "tokens_with_depth": [
      { "token": "일",     "pitch_level": 1, "volume_level": 0 }, # num of left Divide(d=1)
      { "token": "나누기", "pitch_level": 1, "volume_level": 0 }, # left Divide(d=1)

      { "token": "이",     "pitch_level": 3, "volume_level": 0 }, # 2 in Add(2,3), d=2
      { "token": "더하기", "pitch_level": 3, "volume_level": 0 }, # Add(2,3)
      { "token": "삼",     "pitch_level": 3, "volume_level": 0 }, # 3

      { "token": "사",     "pitch_level": 3, "volume_level": 1 }, # 4 in Add(4,5), d=2
      { "token": "더하기", "pitch_level": 3, "volume_level": 0 }, # Add(4,5)
      { "token": "오",     "pitch_level": 3, "volume_level": 0 }, # 5

      # { "token": "곱",     "pitch_level": 0, "volume_level": 0 }, # 5

      { "token": "육",     "pitch_level": 2, "volume_level": 0 }, # 6 in Add(6,7), d=2 (first token of right child → vol=1)
      { "token": "더하기", "pitch_level": 2, "volume_level": 0 }, # Add(6,7)
      { "token": "칠",     "pitch_level": 2, "volume_level": 0 }, # 7

      { "token": "팔",     "pitch_level": 2, "volume_level": 1 }, # 8 in Add(8,9), d=2
      { "token": "더하기", "pitch_level": 2, "volume_level": 0 }, # Add(8,9)
      { "token": "구",     "pitch_level": 2, "volume_level": 0 }, # 9

      { "token": "나누기", "pitch_level": 1, "volume_level": 0 }, # right Divide(d=1)
      { "token": "이",     "pitch_level": 1, "volume_level": 0 }  # denom 2
    ]
  },

  {
    "id": 4,
    "expression_object": Lim(Value('n'), Infty(), Slash(Power(Add(Value(1), Frac(Value(1), Value('n'))), Add(Value('n'), Value(1))), Value('n'))),
    "expression": "lim_(n→∞) (1+1/n)^(n+1) / n",
    "text_ko": "리미트 엔 이 무한 으로 갈 때 엔 분의 일 더하기 엔 분의 일 의 엔 더하기 일 제곱",
    "tokens_with_depth": [
      { "token": "리미트",   "pitch_level": 0, "volume_level": 0 }, # Lim
      { "token": "엔",       "pitch_level": 0, "volume_level": 0 }, # var
      { "token": "이",       "pitch_level": 0, "volume_level": 0 },
      { "token": "무한",     "pitch_level": 0, "volume_level": 0 }, # Infty
      { "token": "으로", "pitch_level": 0, "volume_level": 0 },
      { "token": "갈", "pitch_level": 0, "volume_level": 0 },
      { "token": "때", "pitch_level": 0, "volume_level": 0 },

      { "token": "엔",       "pitch_level": 1, "volume_level": 0 }, # denom n in Slash(d=1)
      { "token": "분의",     "pitch_level": 1, "volume_level": 0 }, # Slash

      { "token": "일",       "pitch_level": 3, "volume_level": 0 }, # 1 in Add base(d=3)
      { "token": "더하기",   "pitch_level": 3, "volume_level": 0 }, # Add(d=3)

      { "token": "엔",       "pitch_level": 3, "volume_level": 0 }, # denom n in Frac(d=3)
      { "token": "분의",     "pitch_level": 3, "volume_level": 0 }, # Frac
      { "token": "일",       "pitch_level": 3, "volume_level": 0 }, # num 1

      { "token": "의",       "pitch_level": 2, "volume_level": 0 }, # Power(d=2)

      { "token": "엔",       "pitch_level": 2, "volume_level": 0 }, # n in Add exponent(d=2)
      { "token": "더하기",   "pitch_level": 2, "volume_level": 0 },
      { "token": "일",       "pitch_level": 2, "volume_level": 0 },

      { "token": "제곱",     "pitch_level": 2, "volume_level": 0 }  # Power
    ]
  },

  {
    "id": 5,
    "expression_object": Sum(Slash(Add(ImplicitMult(Value(2), Value('n')), Value(1)), ImplicitMult(Value('n'), Add(Value('n'), Value(1)))), Value('n'), Value(1), Infty()),
    "expression": "∑_{n=1}^{∞} (2n+1)/(n(n+1))",
    "text_ko": "시그마 엔 은 일 부터 무한 까지 엔 엔 더하기 일 분의 이 엔 더하기 일",
    "tokens_with_depth": [
      { "token": "시그마",   "pitch_level": 0, "volume_level": 0 }, # Sum
      { "token": "엔",       "pitch_level": 0, "volume_level": 0 }, # var
      { "token": "은",       "pitch_level": 0, "volume_level": 0 },
      { "token": "일",   "pitch_level": 0, "volume_level": 0 }, # start
      { "token": "부터", "pitch_level": 0, "volume_level": 0 },
      { "token": "무한", "pitch_level": 0, "volume_level": 0 }, # end
      { "token": "까지", "pitch_level": 0, "volume_level": 0 },     

      { "token": "엔",       "pitch_level": 2, "volume_level": 0 }, # n in denom Mult(d=2)

      { "token": "엔",       "pitch_level": 3, "volume_level": 0 }, # n in Add(n,1)(d=3)
      { "token": "더하기",   "pitch_level": 3, "volume_level": 0 },
      { "token": "일",       "pitch_level": 3, "volume_level": 0 },

      { "token": "분의",     "pitch_level": 1, "volume_level": 0 }, # Slash(d=1)

      { "token": "이",       "pitch_level": 2, "volume_level": 0 }, # 2 in num Mult(d=2)
      { "token": "엔",       "pitch_level": 2, "volume_level": 0 },

      { "token": "더하기",   "pitch_level": 2, "volume_level": 0 }, # Add(num)(d=2)
      { "token": "일",       "pitch_level": 2, "volume_level": 0 }  # 1
    ]
  },

  {
    "id": 6,
    "expression_object": Integral(Value(0), Value(1), Add(Power(Value('x'), Value(2)), Slash(Value(1), Value('x'))), Value('x')),
    "expression": "∫_0^1 (x² + 1/x) dx",
    "text_ko": "인티그럴 영 부터 일 까지 엑스 제곱 더하기 엑스 분의 일 디 엑스",
    "tokens_with_depth": [
      { "token": "인티그럴", "pitch_level": 0, "volume_level": 0 }, # Integral
      { "token": "영",   "pitch_level": 0, "volume_level": 0 }, # lower      
      { "token": "부터", "pitch_level": 0, "volume_level": 0 },
      { "token": "일",   "pitch_level": 0, "volume_level": 0 }, # upper
      { "token": "까지", "pitch_level": 0, "volume_level": 0 },

      { "token": "엑스",     "pitch_level": 1, "volume_level": 0 }, # x in Power(d=1)
      { "token": "제곱",     "pitch_level": 1, "volume_level": 0 }, # Power
      { "token": "더하기",   "pitch_level": 1, "volume_level": 0 }, # Add(d=1)

      { "token": "엑스",     "pitch_level": 1, "volume_level": 0 }, # denom x in Slash(d=1)
      { "token": "분의",     "pitch_level": 1, "volume_level": 0 }, # Slash
      { "token": "일",       "pitch_level": 1, "volume_level": 0 }, # num 1

      { "token": "디",       "pitch_level": 0, "volume_level": 0 }, # from Integral/var 영역
      { "token": "엑스",     "pitch_level": 0, "volume_level": 0 }  # var x (non-operator)
    ]
  },

  {
    "id": 7,
    "expression_object": SQRT(Add(Value(1), SQRT(Add(Value(2), Value(3))))),
    "expression": "√(1 + √(2+3))",
    "text_ko": "루트 일 더하기 루트 이 더하기 삼",
    "tokens_with_depth": [
      { "token": "루트",   "pitch_level": 0, "volume_level": 0 }, # outer SQRT(d=0)
      { "token": "일",     "pitch_level": 1, "volume_level": 0 }, # 1 in Add(d=1)
      { "token": "더하기", "pitch_level": 1, "volume_level": 0 }, # Add(d=1)

      { "token": "루트",   "pitch_level": 1, "volume_level": 0 }, # inner SQRT(d=1)

      { "token": "이",     "pitch_level": 2, "volume_level": 0 }, # 2 in inner Add(d=2)
      { "token": "더하기", "pitch_level": 2, "volume_level": 0 }, # inner Add
      { "token": "삼",     "pitch_level": 2, "volume_level": 0 }  # 3
    ]
  },

  {
    "id": 8,
    "expression_object": Prob(SetIn(Value('X'), SetCup(Value('A'), Value('B'))), SetCap(Value('A'), Value('B'))),
    "expression": "P(X ∈ A ∪ B | A ∩ B)",
    "text_ko": "확률 피 대문자 엑스 는 대문자 에이 합집합 대문자 비 의 원소 바 대문자 에이 교집합 대문자 비",
    "tokens_with_depth": [
      { "token": "확률",   "pitch_level": 0, "volume_level": 0 }, # Prob
      { "token": "피",     "pitch_level": 0, "volume_level": 0 },

      { "token": "대문자 엑스", "pitch_level": 1, "volume_level": 0 }, # X, SetIn(d=1)
      { "token": "는", "pitch_level": 1, "volume_level": 0 }, # X, SetIn(d=1)
      { "token": "대문자 에이",   "pitch_level": 1, "volume_level": 0 }, # A in SetCup(d=1)
      { "token": "합집합", "pitch_level": 1, "volume_level": 0 }, # SetCup
      { "token": "대문자 비",   "pitch_level": 1, "volume_level": 0 }, # B
      {"token": "의",       "pitch_level": 1, "volume_level": 0},      
      { "token": "원소",   "pitch_level": 1, "volume_level": 0 },

      { "token": "바",     "pitch_level": 0, "volume_level": 0 }, # 조건

      { "token": "대문자 에이",   "pitch_level": 1, "volume_level": 0 }, # A in SetCap(d=1)
      { "token": "교집합", "pitch_level": 1, "volume_level": 0 }, # SetCap
      { "token": "대문자 비",     "pitch_level": 1, "volume_level": 0 }  # B
    ]
  },

  {
    "id": 9,
    "expression_object": InnerProduct(Add(Vec(Value('a')), Vec(Value('b'))), Sub(Vec(Value('c')), Vec(Value('d')))),
    "expression": "(\vce{a}+\vec{b}) · (\vec{c}-\vec{d})",
    "text_ko": "벡터 에이 더하기 벡터 비 내적 벡터 씨 빼기 벡터 디",
    "tokens_with_depth": [
      { "token": "벡터",   "pitch_level": 1, "volume_level": 0 }, # Vec(d=1)
      { "token": "에이",   "pitch_level": 1, "volume_level": 0 }, # a in Add(d=1)
      { "token": "더하기", "pitch_level": 1, "volume_level": 0 }, # Add
      { "token": "벡터",   "pitch_level": 1, "volume_level": 0 }, # Vec(d=1)
      { "token": "비",     "pitch_level": 1, "volume_level": 0 }, # b

      { "token": "내적",   "pitch_level": 0, "volume_level": 0 }, # particle

      { "token": "벡터",   "pitch_level": 1, "volume_level": 0 }, # Vec(d=1)
      { "token": "씨",     "pitch_level": 1, "volume_level": 0 }, # c in Sub(d=1)
      { "token": "빼기",   "pitch_level": 1, "volume_level": 0 }, # Sub
      { "token": "벡터",   "pitch_level": 1, "volume_level": 0 }, # Vec(d=0)
      { "token": "디",     "pitch_level": 1, "volume_level": 0 }, # d
    ]
  },

  {
    "id": 10,
    "expression_object": Add(Log(Value(10), Value('x')), Ln(SQRT(Add(Power(Sin(Value('x')), Value(2)), Power(Cos(Value('x')), Value(2)))))),
    "expression": "log_{10} x + ln(√(sin²x + cos²x))",
    "text_ko": "로그 십 의 엑스 더하기 엘엔 루트 사인 제곱 엑스 더하기 코사인 제곱 엑스",
    "tokens_with_depth": [
      { "token": "로그",   "pitch_level": 0, "volume_level": 0 }, # Log(d=0)
      { "token": "십",     "pitch_level": 0, "volume_level": 0 }, # base 10
      { "token": "의",     "pitch_level": 0, "volume_level": 0 },
      { "token": "엑스",   "pitch_level": 0, "volume_level": 0 }, # arg x

      { "token": "더하기", "pitch_level": 0, "volume_level": 0 }, # Add root(d=0)

      { "token": "엘엔",   "pitch_level": 0, "volume_level": 0 }, # Ln(d=0)
      { "token": "루트",   "pitch_level": 0, "volume_level": 0 }, # SQRT(d=0)

      { "token": "사인",   "pitch_level": 1, "volume_level": 0 }, # Sin(d=1)
      { "token": "제곱",   "pitch_level": 1, "volume_level": 0 }, # Power(d=1)
      { "token": "엑스",   "pitch_level": 1, "volume_level": 0 }, # x in Sin

      { "token": "더하기", "pitch_level": 1, "volume_level": 0 }, # inner Add(d=1)

      { "token": "코사인", "pitch_level": 1, "volume_level": 0 }, # Cos(d=1)
      { "token": "제곱",   "pitch_level": 1, "volume_level": 0 },  # Power(d=1)
      { "token": "엑스",   "pitch_level": 1, "volume_level": 0 }, # x in Cos
    ]
  },
  {
      "id": 11,
      "expression_object": SetBuilder(Value('x'), Less(Value(0), Leq(Value('x'), Value(1)))),
      "expression": "{ x | 0 < x ≤ 1 }",
      "text_ko": "집합 엑스 바 영 은 엑스 보다 작고 엑스 는 일 이하이다",
      "tokens_with_depth": [
          {"token": "집합",           "pitch_level": 0, "volume_level": 0},

          # x (SetBuilder의 var, pitch = d+1 = 1)
          {"token": "엑스",           "pitch_level": 0, "volume_level": 0},
          {"token": "바",             "pitch_level": 0, "volume_level": 0},

          # 조건 Less(0, Leq(x,1)) 전체를 pitch 1로
          {"token": "영",           "pitch_level": 1, "volume_level": 0},
          {"token": "은",       "pitch_level": 1, "volume_level": 0},
          {"token": "엑스",  "pitch_level": 1, "volume_level": 0},
          {"token": "보다",       "pitch_level": 1, "volume_level": 0},          
          {"token": "작고",       "pitch_level": 1, "volume_level": 0},
          {"token": "엑스",         "pitch_level": 1, "volume_level": 0},
          {"token": "는",       "pitch_level": 1, "volume_level": 0},          
          {"token": "일",             "pitch_level": 1, "volume_level": 0},
          {"token": "이하이다",       "pitch_level": 1, "volume_level": 0},
      ],
  },

  {
      "id": 12,
      # 0 < 1/(x+1) < 2/(x-1)
      "expression_object": Less(Value(0), Less(Frac(Value(1), Add(Value('x'), Value(1))), Slash(Value(2), Sub(Value('x'), Value(1))))
      ),
      "expression": "0 < 1/(x+1) < 2/(x-1)",
      "text_ko": "영 은 엑스 더하기 일 분의 일 보다 작고 엑스 더하기 일 분의 일 은 엑스 빼기 일 분의 이 보다 작다",
      "tokens_with_depth": [
          {"token": "영",           "pitch_level": 0, "volume_level": 0},
          {"token": "은",           "pitch_level": 0, "volume_level": 0},          

          # 1/(x+1) : Slash(d=0), b = Add(x,1) → pitch 1
          {"token": "엑스",           "pitch_level": 1, "volume_level": 0},
          {"token": "더하기",         "pitch_level": 1, "volume_level": 0},
          {"token": "일",             "pitch_level": 1, "volume_level": 0},
          {"token": "분의",           "pitch_level": 0, "volume_level": 0},
          {"token": "일",             "pitch_level": 0, "volume_level": 0},

          {"token": "보다",      "pitch_level": 0, "volume_level": 0},
          {"token": "작고",      "pitch_level": 0, "volume_level": 0},

          {"token": "엑스",           "pitch_level": 1, "volume_level": 0},
          {"token": "더하기",         "pitch_level": 1, "volume_level": 0},
          {"token": "일",             "pitch_level": 1, "volume_level": 0},
          {"token": "분의",           "pitch_level": 0, "volume_level": 0},
          {"token": "일",             "pitch_level": 0, "volume_level": 0},
          {"token": "은",      "pitch_level": 0, "volume_level": 0},
          
          # 2/(x-1) : Slash(d=0), b = Sub(x,1) → pitch 1
          {"token": "엑스",           "pitch_level": 1, "volume_level": 0},
          {"token": "빼기",       "pitch_level": 1, "volume_level": 0},
          {"token": "일",             "pitch_level": 1, "volume_level": 0},
          {"token": "분의",           "pitch_level": 0, "volume_level": 0},
          {"token": "이",             "pitch_level": 0, "volume_level": 0},

          {"token": "보다",      "pitch_level": 0, "volume_level": 0},
          {"token": "작다",      "pitch_level": 0, "volume_level": 0}
      ],
  },

  {
      "id": 13,
      # { a_{n+1} }
      "expression_object": Seq(Subscript(Value('a'), Add(Value('n'), Value(1)))),
      "expression": "{a_{n+1}}",
      "text_ko": "수열 에이 엔 더하기 일",
      "tokens_with_depth": [
          {"token": "수열",   "pitch_level": 0, "volume_level": 0},  # Seq(d=0)

          # a (연산자 아님) → pitch 0
          {"token": "에이",   "pitch_level": 0, "volume_level": 0},

          # n+1 (비정수 첨자) → pitch 1, 첫 글자 volume -1
          {"token": "엔",     "pitch_level": 1, "volume_level": -1},
          {"token": "더하기", "pitch_level": 1, "volume_level": 0},
          {"token": "일",     "pitch_level": 1, "volume_level": 0},
      ],
  },

  {
      "id": 14,
      # f(x) = { 1 (x<0), 0 (x=0), -1 (x>0) }
      "expression_object": Eq(Func(Value('f'), [Value('x')]),
          Cases([
              (Value(1), Less(Value('x'), Value(0))),
              (Value(0), Eq(Value('x'), Value(0))),
              (Minus(Value(1)), Greater(Value('x'), Value(0)))
          ])
      ),
      "expression": "f(x) = { 1 (x<0), 0 (x=0), -1 (x>0) }",
      "text_ko": "에프 엑스 는 엑스 가 영 보다 작을 때 일 그리고 엑스 는 영 일 때 영 그리고 엑스 가 영 보다 클 때 마이너스 일",
      "tokens_with_depth": [
          {"token": "에프",           "pitch_level": 0, "volume_level": 0},
          {"token": "엑스",           "pitch_level": 0, "volume_level": 0},
          {"token": "는",           "pitch_level": 0, "volume_level": 0},

          # case 1
          {"token": "엑스",           "pitch_level": 0, "volume_level": 0},
          {"token": "가",           "pitch_level": 0, "volume_level": 0},
          {"token": "영",   "pitch_level": 0, "volume_level": 0},
          {"token": "보다",           "pitch_level": 0, "volume_level": 0},
          {"token": "작을",           "pitch_level": 0, "volume_level": 0},
          {"token": "때",           "pitch_level": 0, "volume_level": 0},
          {"token": "일",               "pitch_level": 0, "volume_level": 0},

          {"token": "그리고",           "pitch_level": 0, "volume_level": 0},

          # case 2
          {"token": "엑스",           "pitch_level": 0, "volume_level": 0},
          {"token": "는",           "pitch_level": 0, "volume_level": 0},
          {"token": "영",          "pitch_level": 0, "volume_level": 0},
          {"token": "일",           "pitch_level": 0, "volume_level": 0},
          {"token": "때",           "pitch_level": 0, "volume_level": 0},
          {"token": "영",               "pitch_level": 0, "volume_level": 0},

          {"token": "그리고",           "pitch_level": 0, "volume_level": 0},

          # case 3
          {"token": "엑스",           "pitch_level": 0, "volume_level": 0},
          {"token": "가",           "pitch_level": 0, "volume_level": 0},
          {"token": "영",     "pitch_level": 0, "volume_level": 0},
          {"token": "보다",           "pitch_level": 0, "volume_level": 0},
          {"token": "클",           "pitch_level": 0, "volume_level": 0},
          {"token": "때",           "pitch_level": 0, "volume_level": 0},
          {"token": "마이너스",         "pitch_level": 0, "volume_level": 0},
          {"token": "일",               "pitch_level": 0, "volume_level": 0},
      ],
  },

  {
      "id": 15,
      # AB ⟂ CD ∥ EF
      "expression_object": Perp(
          Line(Value('A'), Value('B')),
          Paral(
              Line(Value('C'), Value('D')),
              Line(Value('E'), Value('F'))
          )
      ),
      "expression": "AB ⟂ CD ∥ EF",
      "text_ko": "직선 대문자 에이 대문자 비 와 직선 대문자 씨 대문자 디 는 수직이고 직선 대문자 씨 대문자 디 와 직선 대문자 이 대문자 에프 는 평행이다",
      "tokens_with_depth": [
          {"token": "직선",   "pitch_level": 0, "volume_level": 0},
          {"token": "대문자 에이",   "pitch_level": 0, "volume_level": 0},
          {"token": "대문자 비",     "pitch_level": 0, "volume_level": 0},
          {"token": "와",     "pitch_level": 0, "volume_level": 0},

          {"token": "직선",   "pitch_level": 0, "volume_level": 0},
          {"token": "대문자 씨",     "pitch_level": 0, "volume_level": 0},
          {"token": "대문자 디",     "pitch_level": 0, "volume_level": 0},
          {"token": "는",     "pitch_level": 0, "volume_level": 0},

          {"token": "수직이고",   "pitch_level": 0, "volume_level": 0},

          {"token": "직선",   "pitch_level": 0, "volume_level": 0},
          {"token": "대문자 씨",     "pitch_level": 0, "volume_level": 0},
          {"token": "대문자 디",     "pitch_level": 0, "volume_level": 0},
          {"token": "와",     "pitch_level": 0, "volume_level": 0},

          {"token": "직선",   "pitch_level": 0, "volume_level": 0},
          {"token": "대문자 이",     "pitch_level": 0, "volume_level": 0},
          {"token": "대문자 에프",   "pitch_level": 0, "volume_level": 0},
          {"token": "는",     "pitch_level": 0, "volume_level": 0},

          {"token": "평행이다",   "pitch_level": 0, "volume_level": 0},
      ],
  },


  {
      "id": 16,
      # (x + 1/y) / (1/(x + y))
      "expression_object": Frac(
          Add(Value('x'), Frac(Value(1), Value('y'))),
          Slash(Value(1), Add(Value('x'), Value('y')))
          ),
      "expression": "(x + 1/y) / (1/(x + y))",
      "text_ko": "엑스 더하기 와이 분의 일 분의 엑스 더하기 와이 분의 일",
      "tokens_with_depth": [
          # 분모: Slash(1, x+y), d=1
          {"token": "엑스",       "pitch_level": 2, "volume_level": 0},
          {"token": "더하기",     "pitch_level": 2, "volume_level": 0},
          {"token": "와이",       "pitch_level": 2, "volume_level": 0},
          {"token": "분의",       "pitch_level": 1, "volume_level": 0},
          {"token": "일",         "pitch_level": 1, "volume_level": 0},

          # 외부 Slash(d=0)
          {"token": "분의",       "pitch_level": 0, "volume_level": 0},

          # 분자: Add(x, 1/y), d=1
          {"token": "엑스",       "pitch_level": 1, "volume_level": 0},
          {"token": "더하기",     "pitch_level": 1, "volume_level": 0},
          {"token": "와이",       "pitch_level": 1, "volume_level": 0},
          {"token": "분의",       "pitch_level": 1, "volume_level": 0},
          {"token": "일",         "pitch_level": 1, "volume_level": 0},
      ],
  },

  {
      "id": 17,
      # (a+1)(b+1)  → ImplicitMult(Add(a,1), Add(b,1))
      "expression_object": ImplicitMult(Add(Value('a'), Value(1)), Add(Value('b'), Value(1))),
      "expression": "(a+1)(b+1)",
      "text_ko": "에이 더하기 일 비 더하기 일",
      "tokens_with_depth": [
          # 왼쪽 Add(a+1), ImplicitMult(d=0) → pitch 1
          {"token": "에이",       "pitch_level": 1, "volume_level": 0},
          {"token": "더하기",     "pitch_level": 1, "volume_level": 0},
          {"token": "일",         "pitch_level": 1, "volume_level": 0},

          # 오른쪽 Add(b+1), pitch 1, 첫 글자 volume +1 (두 그룹 모두 d+1 조건 만족)
          {"token": "비",         "pitch_level": 1, "volume_level": 1},
          {"token": "더하기",     "pitch_level": 1, "volume_level": 0},
          {"token": "일",         "pitch_level": 1, "volume_level": 0},
      ],
  },

  {
      "id": 18,
      # d²y/dx² + dy/dx − 1 = 0
      "expression_object": Eq(Add(Add(Diff(Value('y'), Value('x'), Value(2)),Diff(Value('y'), Value('x'), Value(1))), Minus(Value(1))
          ),
          Value(0)
      ),
      "expression": "d²y/dx² + dy/dx + −(1) = 0",
      "text_ko": "디 제곱 와이 디 엑스 제곱 더하기 디 와이 디 엑스 더하기 마이너스 일 은 영",
      "tokens_with_depth": [
          {"token": "디",       "pitch_level": 0, "volume_level": 0},
          {"token": "제곱",     "pitch_level": 0, "volume_level": 0},
          {"token": "와이",     "pitch_level": 0, "volume_level": 0},
          {"token": "디",       "pitch_level": 0, "volume_level": 0},
          {"token": "엑스",     "pitch_level": 0, "volume_level": 0},
          {"token": "제곱",     "pitch_level": 0, "volume_level": 0},

          {"token": "더하기",   "pitch_level": 0, "volume_level": 0},

          {"token": "디",       "pitch_level": 0, "volume_level": 0},
          {"token": "와이",     "pitch_level": 0, "volume_level": 0},
          {"token": "디",       "pitch_level": 0, "volume_level": 0},
          {"token": "엑스",     "pitch_level": 0, "volume_level": 0},

          {"token": "더하기", "pitch_level": 0, "volume_level": 0},
          {"token": "마이너스", "pitch_level": 0, "volume_level": 0},
          {"token": "일",     "pitch_level": 0, "volume_level": 0},
          {"token": "은",     "pitch_level": 0, "volume_level": 0},
          {"token": "영",       "pitch_level": 0, "volume_level": 0},
      ],
  },

  {
      "id": 19,
      # P(A | B ∩ C) = P(A ∩ B ∩ C) / P(B ∩ C)
      "expression_object": Eq(
          Prob(
              Value('A'),
              SetCap(Value('B'), Value('C'))
          ),
          Slash(
              Prob(
                  SetCap(
                      SetCap(Value('A'), Value('B')),
                      Value('C')
                  )
              ),
              Prob(
                  SetCap(Value('B'), Value('C'))
              )
          )
      ),
      "expression": "P(A | B ∩ C) = P(A ∩ B ∩ C) / P(B ∩ C)",
      "text_ko": "확률 피 대문자 에이 바 대문자 비 교집합 대문자 씨 는 확률 피 대문자 비 교집합 대문자 씨 분의 확률 피 대문자 에이 교집합 대문자 비 교집합 대문자 씨",
      "tokens_with_depth": [
          # 좌변: Prob(A | B∩C), Eq의 d = 0 → 전부 pitch 0
          {"token": "확률",     "pitch_level": 0, "volume_level": 0},
          {"token": "피",       "pitch_level": 0, "volume_level": 0},
          {"token": "대문자 에이",     "pitch_level": 0, "volume_level": 0},
          {"token": "바",       "pitch_level": 0, "volume_level": 0},
          {"token": "대문자 비",       "pitch_level": 1, "volume_level": 0},
          {"token": "교집합",   "pitch_level": 1, "volume_level": 0},
          {"token": "대문자 씨",       "pitch_level": 1, "volume_level": 0},

          # Eq: 왼쪽 전체 + “는” + 오른쪽
          {"token": "는",       "pitch_level": 0, "volume_level": 0},

          # 우변: Slash(분자, 분모), d = 0
          # 분모 b = Prob(B∩C), 연산자이므로 Slash 규칙에 의해 pitch = d+1 = 1
          {"token": "확률",     "pitch_level": 1, "volume_level": 0},
          {"token": "피",       "pitch_level": 1, "volume_level": 0},
          {"token": "대문자 비",       "pitch_level": 2, "volume_level": 0},
          {"token": "교집합",   "pitch_level": 2, "volume_level": 0},
          {"token": "대문자 씨",       "pitch_level": 2, "volume_level": 0},

          # Slash 자체 토큰
          {"token": "분의",     "pitch_level": 0, "volume_level": 0},

          # 분자 a = Prob(A∩B∩C), 역시 연산자 → Slash 규칙상 pitch = 1
          {"token": "확률",     "pitch_level": 1, "volume_level": 0},
          {"token": "피",       "pitch_level": 1, "volume_level": 0},

          # 내부 SetCap(A,B), d = 2
          {"token": "대문자 에이",     "pitch_level": 2, "volume_level": 0},
          {"token": "교집합",   "pitch_level": 2, "volume_level": 0},
          {"token": "대문자 비",       "pitch_level": 2, "volume_level": 0},

          # 바깥 SetCap의 '교집합', d = 2
          {"token": "교집합",   "pitch_level": 2, "volume_level": 0},

          # 오른쪽 피연산자 C, d = 2
          {"token": "대문자 씨",       "pitch_level": 2, "volume_level": 0},
      ],
  },
  {
      "id": 20,
      # p \\leftrightarrow q
      "expression_object": Biconditional(Value('p'), Value('q')),
      "expression": "p \\\leftrightarrow q",
      "text_ko": "피 는 큐 의 필요충분조건이다",
      "tokens_with_depth": [
          {"token": "피",       "pitch_level": 0, "volume_level": 0},
          {"token": "는",       "pitch_level": 0, "volume_level": 0},
          {"token": "큐",       "pitch_level": 0, "volume_level": 0},
          {"token": "의",       "pitch_level": 0, "volume_level": 0},
          {"token": "필요충분조건이다", "pitch_level": 0, "volume_level": 0},
      ],
  },
  {
      "id": 21,
      # p \\iff q
      "expression_object": Iff(Value('p'), Value('q')),
      "expression": "p \\\iff q",
      "text_ko": "참으로 피 는 큐 의 필요충분조건이다",
      "tokens_with_depth": [
          {"token": "참으로",   "pitch_level": 0, "volume_level": 0},
          {"token": "피",       "pitch_level": 0, "volume_level": 0},
          {"token": "는",       "pitch_level": 0, "volume_level": 0},
          {"token": "큐",       "pitch_level": 0, "volume_level": 0},
          {"token": "의",       "pitch_level": 0, "volume_level": 0},
          {"token": "필요충분조건이다", "pitch_level": 0, "volume_level": 0},
      ],
  },
]

def _tokens_to_dict_list(tokens: List[tuple]) -> List[Dict[str, Any]]:
    """
    expression_to_tokens_with_pitch 가 반환하는
    [(token, pitch, volume), ...] 를

    [{"token": str, "pitch_level": int, "volume_level": int}, ...] 로 변환
    """
    return [
        {
            "token": t,
            "pitch_level": p,
            "volume_level": v
        }
        for (t, p, v) in tokens
    ]


def _format_token_list(token_dicts: List[Dict[str, Any]]) -> str:
    """
    디버깅용 예쁜 문자열.
    """
    parts = []
    for td in token_dicts:
        parts.append(
            f"{td['token']} (p={td['pitch_level']}, v={td['volume_level']})"
        )
    return "[ " + ", ".join(parts) + " ]"


def test_grouping_pitch_cases():
    """
    전역 변수 grouping_pitch_test_cases 에 대해
    expression_to_tokens_with_pitch 결과가 기대값과 일치하는지 검사.
    - token / pitch_level / volume_level 모두 비교
    - tokens 를 공백으로 join 해서 text_ko 도 같이 검증
    - PASS 인 경우: Repr, 대체 텍스트, 토큰+피치/볼륨 상세 출력
    - 마지막에 전체 통과 개수 / 전체 케이스 개수 출력
    """
    total_cases = len(grouping_pitch_test_cases)
    passed_count = 0

    for case in grouping_pitch_test_cases:
        cid = case["id"]
        expr_obj = case["expression_object"]
        expected_tokens = case["tokens_with_depth"]
        expected_text_ko = case["text_ko"]

        # 실제 토큰 생성
        actual_tokens_raw = expression_to_tokens_with_pitch(expr_obj)
        actual_tokens = _tokens_to_dict_list(actual_tokens_raw)

        # 1) 길이 비교
        if len(actual_tokens) != len(expected_tokens):
            print(f"[FAIL] case #{cid}: token length mismatch")
            print(f"  expression: {case['expression']}")
            print(f"  expected length: {len(expected_tokens)}")
            print(f"  actual   length: {len(actual_tokens)}")
            print("  expected:", _format_token_list(expected_tokens))
            print("  actual  :", _format_token_list(actual_tokens))
            continue

        # 2) 각 위치별 (token, pitch_level, volume_level) 비교
        case_ok = True
        for i, (exp, act) in enumerate(zip(expected_tokens, actual_tokens)):
            if (
                exp["token"] != act["token"] or
                exp["pitch_level"] != act["pitch_level"] or
                exp["volume_level"] != act["volume_level"]
            ):
                if case_ok:
                    print(f"[FAIL] case #{cid}: token mismatch at index {i}")
                    print(f"  expression: {case['expression']}")
                case_ok = False
                print(
                    f"  idx {i}: "
                    f"expected=({exp['token']!r}, p={exp['pitch_level']}, v={exp['volume_level']}) "
                    f"actual=({act['token']!r}, p={act['pitch_level']}, v={act['volume_level']})"
                )

        # 3) text_ko 검증 (토큰 문자열을 공백으로 join)
        actual_text_ko = " ".join(t["token"] for t in actual_tokens)
        if actual_text_ko != expected_text_ko:
            if case_ok:
                print(f"[FAIL] case #{cid}: text_ko mismatch")
                print(f"  expression: {case['expression']}")
            case_ok = False
            print(f"  expected text_ko: {expected_text_ko!r}")
            print(f"  actual   text_ko: {actual_text_ko!r}")

        if case_ok:
            passed_count += 1
            # 예쁜 PASS 출력
            bar = "=" * 100
            print(f"{bar} 예시 {cid}: {case['expression']} {bar}")
            print(f"수식: {case['expression']}")
            print(f"Repr: {repr(expr_obj)}")

            # 대체 텍스트 (expression_to_korean 이 있다고 가정)
            try:
                alt_text = expression_to_korean(expr_obj)
            except NameError:
                # 함수가 없으면 스킵
                alt_text = "(expression_to_korean 미정의)"

            print(f"대체 텍스트: {alt_text}")

            print("토큰+피치/볼륨:")
            print(f"  텍스트: {actual_text_ko}")
            for td in actual_tokens:
                print(f"  {td['token']}: pitch={td['pitch_level']}, volume={td['volume_level']}")
        else:
            # 이미 FAIL 정보 출력됨
            pass

    print(f"\n전체 통과: {passed_count}/{total_cases}")

if __name__ == "__main__":
    test_grouping_pitch_cases()
