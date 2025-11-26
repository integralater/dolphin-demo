from Expression_Syntax import *
from LaTeX_Parser import *
from speech_synthesizer import *
from gtts_expr_audio_pitch import *
from audio_pitch import *
from grouping_pitch import *

# =========================================================
# Token
# =========================================================

class Token:
    __slots__ = ("kind", "value", "start", "end")

    def __init__(self, kind: str, value: str, start: int = -1, end: int = -1):
        # kind: 'CMD', 'ID', 'NUM', 'SYM'
        self.kind = kind
        self.value = value
        self.start = start
        self.end = end

    def __repr__(self):
        return f"Token({self.kind!r}, {self.value!r})"


def _token_triggers_operator_context(token: Optional[Token]) -> bool:
    """대괄호가 연산 그룹으로 해석되는 문맥인지 판단 (모든 이항 중위 연산자 포함)"""
    if token is None:
        return False
    
    # 피연산자: 숫자, 변수, 닫는 괄호, 팩토리얼 (연산자 뒤에 올 수 있음)
    if token.kind in ("NUM", "ID"):
        return True
    if token.kind == "SYM" and token.value in (")", "]", "!"):
        return True
    
    # 이항 중위 연산자
    # 1. 산술 연산자
    if token.kind == "SYM" and token.value in ("+", "-", "/", ":"):
        return True
    if token.kind == "CMD" and token.value in (r"\times", r"\cdot", r"\div", r"\pm", r"\mp"):
        return True
    
    # 2. 집합 연산자
    if token.kind == "CMD" and token.value in (r"\cup", r"\cap"):
        return True
    
    # 3. 비교/관계 연산자
    if token.kind == "SYM" and token.value in ("=", "<", ">"):
        return True
    if token.kind == "CMD" and token.value in (
        r"\leq", r"\geq", r"\neq",
        r"\approx", r"\approxeq", r"\fallingdotseq"
    ):
        return True
    
    # 4. 집합 관계 연산자
    if token.kind == "CMD" and token.value in (
        r"\in", r"\notin",
        r"\subset", r"\subseteq", r"\supset", r"\supseteq"
    ):
        return True
    
    # 5. 논리 연산자
    if token.kind == "CMD" and token.value in (
        r"\to", r"\rightarrow", r"\leftarrow", r"\gets",
        r"\Rightarrow", r"\Leftarrow",
        r"\leftrightarrow", r"\iff", r"\Leftrightarrow"
    ):
        return True
    
    # 6. 기하 관계 연산자
    if token.kind == "CMD" and token.value in (
        r"\equiv", r"\sim", r"\parallel", r"\perp"
    ):
        return True
    
    return False


# =========================================================
# Parser
# =========================================================

class LatexParser:
    def __init__(self, src: str):
        self.src = self._preprocess(src)
        self.tokens = self._lex(self.src)
        self.pos = 0
        self.limit = len(self.tokens)

    _GREEK_CMD_MAP = {
        r'\alpha': 'α',
        r'\beta': 'β',
        r'\gamma': 'γ',
        r'\delta': 'δ',
        r'\epsilon': 'ε',
        r'\zeta': 'ζ',
        r'\eta': 'η',
        r'\theta': 'θ',
        r'\iota': 'ι',
        r'\kappa': 'κ',
        r'\lambda': 'λ',
        r'\mu': 'μ',
        r'\nu': 'ν',
        r'\xi': 'ξ',
        r'\pi': 'π',
        r'\rho': 'ρ',
        r'\sigma': 'σ',
        r'\tau': 'τ',
        r'\upsilon': 'υ',
        r'\phi': 'φ',
        r'\chi': 'χ',
        r'\psi': 'ψ',
        r'\omega': 'ω',
        r'\Gamma': 'Γ',
        r'\Delta': 'Δ',
        r'\Theta': 'Θ',
        r'\Lambda': 'Λ',
        r'\Xi': 'Ξ',
        r'\Pi': 'Π',
        r'\Sigma': 'Σ',
        r'\Upsilon': 'Υ',
        r'\Phi': 'Φ',
        r'\Psi': 'Ψ',
        r'\Omega': 'Ω',
    }

    # -----------------------------------------------------
    # 전처리
    # -----------------------------------------------------

    def _preprocess(self, s: str) -> str:
        s = s.strip()

        def _strip_size_cmd(text: str, cmd: str) -> str:
            result_chars = []
            i = 0
            n = len(text)
            cmd_len = len(cmd)
            while i < n:
                if text.startswith(cmd, i):
                    j = i + cmd_len
                    next_is_alpha = j < n and text[j].isalpha()
                    if next_is_alpha:
                        result_chars.append(cmd)
                    i = j
                    continue
                result_chars.append(text[i])
                i += 1
            return "".join(result_chars)

        s = _strip_size_cmd(s, r"\left")
        s = _strip_size_cmd(s, r"\right")
        
        # 띄어쓰기 명령어 제거 (spacing commands)
        # 주의: \\ (줄바꿈)은 보존해야 하므로 순서가 중요
        spacing_cmds = [r"\!", r"\,", r"\:", r"\;", r"\quad", r"\qquad", 
                       r"\enspace", r"\smallskip", r"\medskip", r"\bigskip"]
        for cmd in spacing_cmds:
            s = s.replace(cmd, "")
        
        # "\ " (백슬래시+공백)만 제거하되, "\\" (줄바꿈)은 보존
        # 먼저 "\\"를 임시 토큰으로 치환 -> "\ " 제거 -> "\\" 복원
        s = s.replace(r"\\", "<!NEWLINE!>")
        s = s.replace(r"\ ", "")
        s = s.replace("<!NEWLINE!>", r"\\")
        
        return s

    # ----------------------------------------------------- 
    # Lexer
    # -----------------------------------------------------

    def _lex(self, s: str) -> List[Token]:
        tokens: List[Token] = []
        i, n = 0, len(s)

        while i < n:
            ch = s[i]

            if ch.isspace():
                i += 1
                continue

            if ch == "\\":
                start = i
                j = i + 1

                # \{, \}
                if j < n and s[j] in "{}":
                    cmd = "\\" + s[j]
                    tokens.append(Token("CMD", cmd, start, j + 1))
                    i = j + 1
                    continue

                if j < n and not s[j].isalpha():
                    cmd = s[i:j+1]
                    tokens.append(Token("CMD", cmd, start, j + 1))
                    i = j + 1
                    continue

                while j < n and s[j].isalpha():
                    j += 1
                cmd = s[i:j]
                tokens.append(Token("CMD", cmd, start, j))
                i = j
                continue

            if ch.isdigit():
                start = i
                j = i + 1
                while j < n and s[j].isdigit():
                    j += 1
                tokens.append(Token("NUM", s[i:j], start, j))
                i = j
                continue

            if ch.isalpha():
                start = i
                j = i + 1
                while j < n and s[j].isalpha():
                    j += 1
                tokens.append(Token("ID", s[i:j], start, j))
                i = j
                continue

            tokens.append(Token("SYM", ch, i, i + 1))
            i += 1

        return tokens

    # -----------------------------------------------------
    # 토큰 유틸
    # -----------------------------------------------------

    def peek(self, k: int = 0) -> Optional[Token]:
        idx = self.pos + k
        if 0 <= idx < self.limit:
            return self.tokens[idx]
        return None

    def consume(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def match(self, kind: str, value: Optional[str] = None) -> bool:
        tok = self.peek()
        if tok is None:
            return False
        if tok.kind != kind:
            return False
        if value is not None and tok.value != value:
            return False
        return True

    def expect(self, kind: str, value: Optional[str] = None) -> Token:
        if not self.match(kind, value):
            raise ValueError(f"expected {kind} {value}, got {self.peek()}")
        return self.consume()

    def expect_end(self):
        if self.peek() is not None:
            raise ValueError(f"extra tokens at end: {self.tokens[self.pos:]}")

    # =========================================================
    # 최상위
    # =========================================================

    def parse_full(self) -> Expression:
        return self.parse_equation_chain()

    # =========================================================
    # 비율 레벨 (: 연산자 - 가장 낮은 우선순위)
    # =========================================================

    def parse_ratio_chain(self) -> Expression:
        """비율 연산자(:) 파싱"""
        left = self.parse_add_sub()
        ratios: List[Expression] = [left]

        while True:
            tok = self.peek()
            if tok is None or not (tok.kind == "SYM" and tok.value == ":"):
                break

            self.consume()
            next_expr = self.parse_add_sub()
            ratios.append(next_expr)

        if len(ratios) == 1:
            return ratios[0]
        else:
            return Ratio(ratios)

    # =========================================================
    # 관계/서술 레벨
    # =========================================================

    def parse_equation_chain(self) -> Expression:
        left = self.parse_ratio_chain()
        rels: list[tuple[str, Expression]] = []

        while True:
            tok = self.peek()
            if tok is None:
                break

            op = None

            if tok.kind == "SYM" and tok.value == "=":
                op = "eq"
            elif tok.kind == "SYM" and tok.value == "<":
                op = "lt"
            elif tok.kind == "SYM" and tok.value == ">":
                op = "gt"
            elif tok.kind == "CMD" and tok.value == r"\leq":
                op = "leq"
            elif tok.kind == "CMD" and tok.value == r"\geq":
                op = "geq"
            elif tok.kind == "CMD" and tok.value == r"\neq":
                op = "neq"

            elif tok.kind == "CMD" and tok.value == r"\in":
                op = "in"
            elif tok.kind == "CMD" and tok.value == r"\notin":
                op = "notin"
            elif tok.kind == "CMD" and tok.value == r"\subset":
                op = "subset"
            elif tok.kind == "CMD" and tok.value == r"\subseteq":
                op = "subseteq"
            elif tok.kind == "CMD" and tok.value == r"\supset":
                op = "supset"
            elif tok.kind == "CMD" and tok.value == r"\supseteq":
                op = "supseteq"

            elif tok.kind == "CMD" and tok.value in (r"\to", r"\rightarrow"):
                op = "rimpl"
            elif tok.kind == "CMD" and tok.value == r"\Rightarrow":
                op = "rsufficient"
            elif tok.kind == "CMD" and tok.value in (r"\leftarrow", r"\gets"):
                op = "limpl"
            elif tok.kind == "CMD" and tok.value == r"\Leftarrow":
                op = "lsufficient"
            elif tok.kind == "CMD" and tok.value in (r"\leftrightarrow",):
                op = "bicond"
            elif tok.kind == "CMD" and tok.value in (r"\iff", r"\Leftrightarrow"):
                op = "iff"
            elif tok.kind == "CMD" and tok.value == r"\equiv":
                op = "congru"
            elif tok.kind == "CMD" and tok.value == r"\sim":
                op = "sim"
            elif tok.kind == "CMD" and tok.value == r"\approx":
                op = "approx"
            elif tok.kind == "CMD" and tok.value == r"\approxeq":
                op = "approx"
            elif tok.kind == "CMD" and tok.value == r"\fallingdotseq":
                op = "approx"
            elif tok.kind == "CMD" and tok.value == r"\parallel":
                op = "paral"
            elif tok.kind == "CMD" and tok.value == r"\perp":
                op = "perp"
            else:
                break

            self.consume()
            rhs = self.parse_ratio_chain()
            rels.append((op, rhs))

        if not rels:
            return left

        def make_rel_node(op: str, L: Expression, R: Expression) -> Expression:
            if op == "eq":
                return Eq(L, R)
            if op == "lt":
                return Less(L, R)
            if op == "gt":
                return Greater(L, R)
            if op == "leq":
                return Leq(L, R)
            if op == "geq":
                return Geq(L, R)
            if op == "neq":
                return Neq(L, R)
            if op == "in":
                return SetIn(L, R)
            if op == "notin":
                return SetNotIn(L, R)
            if op in ("subset", "subseteq"):
                return SetSub(L, R)
            if op in ("supset", "supseteq"):
                return SetSup(L, R)
            if op == "rimpl":
                return Rimpl(L, R)
            if op == "rsufficient":
                return Rsufficient(L, R)
            if op == "limpl":
                return Limpl(L, R)
            if op == "lsufficient":
                return Lsufficient(L, R)
            if op == "bicond":
                return Biconditional(L, R)
            if op == "iff":
                return Iff(L, R)
            if op == "congru":
                return Congru(L, R)
            if op == "sim":
                return Sim(L, R)
            if op == "approx":
                return Eq(L, About(R))
            if op == "paral":
                return Paral(L, R)
            if op == "perp":
                return Perp(L, R)
            raise ValueError(f"unknown relation op: {op}")

        op_last, rhs_last = rels[-1]
        left_last = rels[-2][1] if len(rels) >= 2 else left
        node = make_rel_node(op_last, left_last, rhs_last)

        for i in range(len(rels) - 2, -1, -1):
            op_i, rhs_i = rels[i]
            L = left if i == 0 else rels[i - 1][1]
            node = make_rel_node(op_i, L, node)

        return node

    # =========================================================
    # 산술 레벨
    # =========================================================

    def _is_set_like(self, expr: Expression) -> bool:
        if isinstance(expr, Value) and isinstance(expr.val, str):
            if len(expr.val) == 1 and expr.val.isupper():
                return True
        if isinstance(expr, (
            SetCup, SetCap,
            SetSub, SetSup, SetNotSub, SetNotSup,
            SetIn, SetNotIn,
            SetBuilder, SetRoster, EmptySet,
            ClosedInterval, OpenInterval,
            ClosedOpenInterval, OpenClosedInterval,
        )):
            return True
        return False

    def parse_add_sub(self) -> Expression:
        left = self.parse_mul_div()

        while True:
            tok = self.peek()
            if tok is None:
                break

            op = None
            if tok.kind == "SYM" and tok.value == "+":
                op = "add"
            elif tok.kind == "SYM" and tok.value == "-":
                op = "sub"
            elif tok.kind == "CMD" and tok.value == r"\pm":
                op = "pm"
            elif tok.kind == "CMD" and tok.value == r"\mp":
                op = "mp"
            else:
                break

            self.consume()
            right = self.parse_mul_div()

            if op == "add":
                left = Add(left, right)
            elif op == "sub":
                left = Sub(left, right)
            elif op == "pm":
                left = PlusMinus(left, right)
            elif op == "mp":
                left = MinusPlus(left, right)

        return left

    _OPERATOR_CMDS = {
        r'\times', r'\cdot', r'\div', r'\pm', r'\mp',
        r'\leq', r'\geq', r'\neq', r'\approx', r'\approxeq', r'\fallingdotseq',
        r'\to', r'\rightarrow', r'\leftarrow', r'\gets',
        r'\Rightarrow', r'\Leftarrow',
        r'\leftrightarrow', r'\iff', r'\Leftrightarrow',
        r'\cup', r'\cap', r'\in', r'\notin',
        r'\subset', r'\subseteq', r'\supset', r'\supseteq',
        r'\equiv', r'\sim', r'\parallel', r'\perp',
    }

    _SET_OPERATOR_CMDS = {
        r'\cap', r'\cup', r'\setminus', r'\sqcap', r'\sqcup',
        r'\subset', r'\subseteq', r'\supset', r'\supseteq',
        r'\subsetneq', r'\supsetneq', r'\in', r'\notin', r'\ni', r'\notni',
    }

    _FACTOR_START_SYMS = {"(", "{", "["}
    _RECURRING_CMD_SET = {r"\dot", r"\bar", r"\overline"}

    def _can_start_factor(self, tok: Token) -> bool:
        if tok.kind in ("NUM", "ID"):
            return True
        if tok.kind == "CMD":
            if tok.value in self._OPERATOR_CMDS:
                return False
            return True
        if tok.kind == "SYM":
            return tok.value in self._FACTOR_START_SYMS
        return False

    def _is_tuple_candidate(self, expr: Expression) -> bool:
        return isinstance(expr, (Point, OpenInterval))

    def _as_point(self, expr: Expression) -> Expression:
        if isinstance(expr, Point):
            return expr
        if isinstance(expr, OpenInterval):
            return Point(None_(), [expr.left, expr.right])
        return expr

    def parse_mul_div(self) -> Expression:
        left = self.parse_implicit_chain()

        while True:
            tok = self.peek()
            if tok is None:
                break

            op = None
            mul_cmd = None

            if tok.kind == "CMD" and tok.value in (r"\times", r"\cdot"):
                op = "mul"
                mul_cmd = tok.value
            elif tok.kind == "CMD" and tok.value == r"\div":
                op = "div"
            elif tok.kind == "SYM" and tok.value == "/":
                op = "slash"
            elif tok.kind == "CMD" and tok.value == r"\cup":
                op = "cup"
            elif tok.kind == "CMD" and tok.value == r"\cap":
                op = "cap"
            else:
                break

            self.consume()
            right = self.parse_implicit_chain()

            if op == "mul":
                if mul_cmd == r"\cdot" and isinstance(left, Vec) and isinstance(right, Vec):
                    left = InnerProduct(left, right)
                elif mul_cmd == r"\cdot" and self._is_tuple_candidate(left) and self._is_tuple_candidate(right):
                    left = InnerProduct(self._as_point(left), self._as_point(right))
                else:
                    left = Mult(left, right)
            elif op == "div":
                left = Divide(left, right)
            elif op == "slash":
                left = Slash(left, right)
            elif op == "cup":
                left = SetCup(left, right)
            elif op == "cap":
                left = SetCap(left, right)

        return left

    def parse_implicit_chain(self) -> Expression:
        terms: List[Expression] = []
        first = self.parse_power_unary()
        terms.append(first)

        while True:
            tok = self.peek()
            if tok is None or not self._can_start_factor(tok):
                break
            terms.append(self.parse_power_unary())

        if len(terms) == 1:
            return terms[0]

        node = terms[0]
        for t in terms[1:]:
            node = ImplicitMult(node, t)
        return node

    # =========================================================
    # 단항/거듭제곱
    # =========================================================

    def parse_power_unary(self) -> Expression:
        unary_ops = []
        while True:
            tok = self.peek()
            if tok is None:
                break
            if tok.kind == "SYM" and tok.value in "+-":
                unary_ops.append(tok.value)
                self.consume()
            elif tok.kind == "SYM" and tok.value == "~":
                unary_ops.append("not")
                self.consume()
            elif tok.kind == "CMD" and tok.value == r"\sim":
                unary_ops.append("not")
                self.consume()
            else:
                break

        node = self.parse_power_core()

        for op in reversed(unary_ops):
            if op == "+":
                node = Plus(node)
            elif op == "-":
                node = Minus(node)
            elif op == "not":
                node = Not(node)

        return node

    def _parse_exponent(self) -> Expression:
        if self.match("SYM", "{"):
            return self._parse_brace_block(as_group=True)

        tok = self.peek()
        if tok is None:
            return None_()

        if tok.kind == "NUM":
            self.consume()
            int_part = tok.value
            number_expr: Expression = Value(int(int_part))
            if self.match("SYM", "."):
                next_tok = self.peek(1)
                if next_tok and next_tok.kind == "NUM":
                    self.consume()  # consume '.'
                    frac_tok = self.consume()
                    number_expr = Value(float(f"{int_part}.{frac_tok.value}"))
                    if (
                        self.match("CMD")
                        and self.peek().value in self._RECURRING_CMD_SET
                    ):
                        recurring_phrase = self._parse_recurring_phrase()
                        return RecurringDecimal(number_expr, recurring_phrase)
                    return number_expr
                elif (
                    next_tok
                    and next_tok.kind == "CMD"
                    and next_tok.value in self._RECURRING_CMD_SET
                ):
                    self.consume()  # consume '.'
                    recurring_phrase = self._parse_recurring_phrase()
                    return RecurringDecimal(number_expr, recurring_phrase)
            return number_expr

        if tok.kind == "ID":
            self.consume()
            name = tok.value
            if len(name) == 1:
                return Value(name)
            node: Expression = Value(name[0])
            for ch in name[1:]:
                node = ImplicitMult(node, Value(ch))
            return node

        return self.parse_atom()

    def parse_power_core(self) -> Expression:
        base = self.parse_postfix_base()
        node = base

        tok = self.peek()
        if tok and tok.kind == "SYM" and tok.value == "^":
            self.consume()
            expo = self._parse_exponent()
            node = Power(node, expo)

            while True:
                tok2 = self.peek()
                if tok2 is None:
                    break

                if tok2.kind == "SYM" and tok2.value == "!":
                    self.consume()
                    node = Factorial(node)
                    continue

                if tok2.kind == "SYM" and tok2.value == "_":
                    self.consume()
                    sub = self._parse_group_or_atom()
                    node = Subscript(node, sub)
                    continue

                if tok2.kind == "SYM" and tok2.value == "'":
                    self.consume()
                    node = Prime(node)
                    continue

                if (
                    tok2.kind == "SYM"
                    and tok2.value == "("
                    and isinstance(node, (Value, Func, FuncInv, Prime))
                    and not isinstance(node, Text)
                ):
                    allow_bar = isinstance(node, Value) and getattr(node, "val", None) == "P"
                    args = self._parse_arg_list(allow_bar_separator=allow_bar)
                    node = Func(node, args)
                    continue

                break

        return node

    def parse_postfix_base(self) -> Expression:
        node = self.parse_atom()

        while True:
            tok = self.peek()
            if tok is None:
                break

            if tok.kind == "SYM" and tok.value == "(":
                if isinstance(node, (Value, Func, FuncInv, Prime)) and not isinstance(node, Text):
                    allow_bar = isinstance(node, Value) and getattr(node, "val", None) == "P"
                    args = self._parse_arg_list(allow_bar_separator=allow_bar)
                    node = Func(node, args)
                    continue
                else:
                    break

            if tok.kind == "SYM" and tok.value == "!":
                self.consume()
                node = Factorial(node)
                continue

            if tok.kind == "SYM" and tok.value == "_":
                self.consume()
                sub = self._parse_group_or_atom()
                node = Subscript(node, sub)
                continue

            if tok.kind == "SYM" and tok.value == "'":
                self.consume()
                node = Prime(node)
                continue

            break

        return node

    # =========================================================
    # 단위 수식 파싱
    # =========================================================

    def parse_unit_expression(self):
        """단위 수식 파싱 (kg · m / s^2 등)"""
        return self._parse_unit_mul_div()

    def _parse_unit_mul_div(self):
        """단위 곱셈/나눗셈 파싱"""
        left = self._parse_unit_power()

        while True:
            tok = self.peek()
            if tok is None:
                break

            if tok.value in (r"\cdot", r"\times", "/"):
                self.consume()
                if tok.value == "/":
                    right = self._parse_unit_power()
                    left = UnitDiv(left, right)
                else:  # \cdot or \times
                    right = self._parse_unit_power()
                    left = UnitMult(left, right)
            elif (tok.kind == "ID") or (tok.kind == "CMD" and tok.value.startswith("\\")):
                # 연속된 단위 토큰은 곱셈으로 처리 (공백으로 구분된 경우)
                right = self._parse_unit_power()
                left = UnitMult(left, right)
            else:
                break

        return left

    def _parse_unit_power(self):
        """단위 거듭제곱 파싱"""
        base = self._parse_unit_atom()

        tok = self.peek()
        if tok and tok.value == "^":
            self.consume()
            # 지수 파싱
            if self.match("SYM", "{"):
                self.consume()
                expo = self.parse_equation_chain()
                self.expect("SYM", "}")
            else:
                # 단일 숫자나 - 부호
                if self.match("SYM", "-"):
                    self.consume()
                    num_tok = self.expect("NUM")
                    expo = Minus(Value(int(num_tok.value)))
                else:
                    num_tok = self.expect("NUM")
                    expo = Value(int(num_tok.value))

            return Power(base, expo)

        return base

    def _parse_unit_atom(self):
        """단위 기본 요소 파싱"""
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of unit expression")

        # 괄호
        if tok.value == "(":
            self.consume()
            expr = self.parse_unit_expression()
            self.expect("SYM", ")")
            return expr

        # 단위 이름 (ID 토큰 또는 CMD 토큰)
        if tok.kind == "ID":
            unit_name = tok.value
            self.consume()
            return Unit(unit_name)
        elif tok.kind == "CMD" and tok.value.startswith("\\"):
            cmd_name = tok.value[1:]  # \ 제거
            self.consume()
            return Unit(cmd_name)

        raise ValueError(f"Unexpected token in unit expression: {tok}")

    # =========================================================
    # ^ / _ 섞인 \int, \sum, \prod
    # =========================================================

    def _parse_cmd_with_limits(self, base_cmd: str) -> Expression:
        lower = None
        upper = None

        self._consume_optional_limits()

        while True:
            tok = self.peek()
            if tok is None or tok.kind != "SYM" or tok.value not in ("^", "_"):
                break

            is_sup = (tok.value == "^")
            self.consume()

            if self.match("SYM", "{"):
                bound = self._parse_brace_block(as_group=True)
            else:
                if self.match("NUM"):
                    num_tok = self.consume()
                    bound = Value(int(num_tok.value))
                elif self.match("ID"):
                    id_tok = self.consume()
                    bound = Value(id_tok.value)
                elif self.match("CMD", r"\infty"):
                    self.consume()
                    bound = Infty()
                else:
                    bound = self.parse_atom()

            if is_sup:
                upper = bound
            else:
                lower = bound

        if lower is None and upper is None:
            return Value(base_cmd)
        if lower is not None and upper is None:
            return Subscript(Value(base_cmd), lower)
        if lower is None and upper is not None:
            return Power(Value(base_cmd), upper)
        return Power(Subscript(Value(base_cmd), lower), upper)

    # =========================================================
    # [ ... ] / [ ... )
    # =========================================================

    def _scan_square_bracket_region(self):
        i = self.pos
        depth_square = 1
        depth_paren = 0
        depth_brace = 0
        saw_paren = False
        saw_brace = False
        has_comma_top = False
        has_top_level_cmd_brace = False

        while i < self.limit:
            t = self.tokens[i]

            if t.kind == "SYM":
                if t.value == "(":
                    depth_paren += 1
                    saw_paren = True
                elif t.value == ")":
                    if depth_paren > 0:
                        depth_paren -= 1
                    elif depth_square == 1 and depth_brace == 0:
                        return i, ")", saw_paren, saw_brace, has_comma_top, has_top_level_cmd_brace
                elif t.value == "{":
                    depth_brace += 1
                    saw_brace = True
                elif t.value == "}":
                    if depth_brace > 0:
                        depth_brace -= 1
                elif t.value == "[":
                    depth_square += 1
                elif t.value == "]":
                    depth_square -= 1
                    if depth_square == 0:
                        return i, "]", saw_paren, saw_brace, has_comma_top, has_top_level_cmd_brace
                elif t.value == ",":
                    # 최상위 [ ... ]에서의 콤마인지 확인
                    if depth_square == 1 and depth_paren == 0 and depth_brace == 0:
                        has_comma_top = True

            elif t.kind == "CMD":
                # 여기서 \{, \}도 brace로 취급
                if t.value == r"\{":
                    depth_brace += 1
                    saw_brace = True
                    if depth_square == 1 and depth_paren == 0 and depth_brace == 1:
                        has_top_level_cmd_brace = True
                elif t.value == r"\}":
                    if depth_brace > 0:
                        depth_brace -= 1

            i += 1

        raise ValueError("unclosed '['")


    # =========================================================
    # \{ ... \} 스캔
    # =========================================================

    def _scan_set_brace_end(self):
        """
        현재 self.pos부터 CMD '\{', '\}'의 중첩을 추적해서
        매칭되는 '\}'의 인덱스를 찾는다.
        동시에 내부 구조를 분석해서:
          - has_paren: 소괄호 존재 여부
          - has_arith: 산술 연산자(+,-,\times,\cdot,\div,/ ) 존재 여부
          - has_comma_top: 최상위(현재 \{...\} 기준)의 콤마 존재 여부
          - has_bar_or_colon_top: 최상위의 '|' 또는 ':' 존재 여부
        를 리턴한다.
        """
        depth = 1          # \{...\} 중첩
        depth_paren = 0    # ( ... ) 중첩
        has_comma_top = False
        has_bar_or_colon_top = False
        has_paren = False
        has_arith = False

        i = self.pos
        while i < self.limit:
            t = self.tokens[i]

            if t.kind == "CMD":
                if t.value == r"\{":
                    depth += 1
                elif t.value == r"\}":
                    depth -= 1
                    if depth == 0:
                        return i, has_paren, has_arith, has_comma_top, has_bar_or_colon_top
                elif t.value in (r"\times", r"\cdot", r"\div"):
                    has_arith = True

            if t.kind == "SYM":
                if t.value == "(":
                    depth_paren += 1
                    has_paren = True
                elif t.value == ")":
                    if depth_paren > 0:
                        depth_paren -= 1
                elif t.value == ",":
                    # 최상위 \{...\}에서의 콤마인지 확인
                    if depth == 1 and depth_paren == 0:
                        has_comma_top = True
                elif t.value in ("|", ":"):
                    if depth == 1 and depth_paren == 0:
                        has_bar_or_colon_top = True
                elif t.value in ("+", "-", "/"):
                    has_arith = True

            i += 1

        raise ValueError("unclosed \\{")

    def _is_set_operator_token(self, tok: Optional[Token]) -> bool:
        if tok is None:
            return False
        if tok.kind == "CMD" and tok.value in self._SET_OPERATOR_CMDS:
            return True
        if tok.kind == "SYM" and tok.value in {"∩", "∪"}:
            return True
        return False

    def _should_force_set_literal(self, open_idx: int, close_idx: int) -> bool:
        prev_idx = open_idx - 1
        prev_tok = self.tokens[prev_idx] if prev_idx >= 0 else None

        next_idx = close_idx + 1
        next_tok = self.tokens[next_idx] if next_idx < len(self.tokens) else None

        if prev_tok is None and next_tok is None:
            return True

        if self._is_set_operator_token(prev_tok) or self._is_set_operator_token(next_tok):
            return True

        if next_tok and next_tok.kind == "SYM" and next_tok.value in {"^", "_"}:
            return True

        return False


    # =========================================================
    # Atom
    # =========================================================

    def parse_atom(self) -> Expression:
        tok = self.peek()
        if tok is None:
            raise ValueError("unexpected end in atom")

        # 숫자
        if tok.kind == "NUM":
            self.consume()
            literal = tok.value
            has_decimal_point = False
            frac_digits = ""

            if self.match("SYM", "."):
                has_decimal_point = True
                self.consume()
                while self.match("NUM"):
                    part = self.consume().value
                    frac_digits += part
                if frac_digits:
                    literal = f"{literal}.{frac_digits}"

            if has_decimal_point and frac_digits:
                number_expr: Expression = Value(literal)
            else:
                # 선행 0 보존을 위해 문자열로 유지
                # 단, 순수 정수이고 선행 0이 없는 경우만 int로 변환
                if literal.startswith('0') and len(literal) > 1:
                    number_expr = Value(literal)  # "0001" 등 선행 0 보존
                else:
                    number_expr = Value(int(literal))  # "123" -> 123

            recurring_parts: List[Expression] = []
            recurring_active = False

            while True:
                peek_tok = self.peek()
                if (
                    peek_tok
                    and peek_tok.kind == "CMD"
                    and peek_tok.value in self._RECURRING_CMD_SET
                ):
                    recurring_active = True
                    phrase = self._parse_recurring_phrase()
                    recurring_parts.extend(_explode_inline_parts(phrase.parts))
                    continue

                if recurring_active and peek_tok:
                    if peek_tok.kind == "NUM":
                        digits = self.consume().value
                        for ch in digits:
                            recurring_parts.append(Value(ch))
                        continue
                    if peek_tok.kind == "ID":
                        ident = self.consume().value
                        for ch in ident:
                            recurring_parts.append(Value(ch))
                        continue

                break

            if recurring_parts:
                return RecurringDecimal(number_expr, InlinePhrase(recurring_parts))

            return number_expr

        # ID
        if tok.kind == "ID":
            self.consume()
            name = tok.value

            if name == "e":
                return EulerNum()

            if name == "n" and self.match("SYM", "("):
                self.consume()
                inner = self.parse_equation_chain()
                self.expect("SYM", ")")
                if self._is_set_like(inner):
                    return SetNum(inner)
                return ImplicitMult(Value("n"), inner)

            if len(name) == 1 and self.match("SYM", "("):
                allow_bar = name == "P"
                args = self._parse_arg_list(allow_bar_separator=allow_bar)
                return Func(Value(name), args)

            if len(name) == 2 and name.isupper():
                return Segment(Value(name[0]), Value(name[1]))

            if len(name) == 1:
                return Value(name)
            else:
                node: Expression = Value(name[0])
                for ch in name[1:]:
                    node = ImplicitMult(node, Value(ch))
                return node

        # ( ... )
        if tok.kind == "SYM" and tok.value == "(":
            self.consume()
            first = self.parse_equation_chain()
            elems = [first]

            if not self.match("SYM", ","):
                self.expect("SYM", ")")
                return first

            while self.match("SYM", ","):
                self.consume()
                elems.append(self.parse_equation_chain())

            if not self.match("SYM") or self.peek().value not in (")", "]"):
                raise ValueError(f"expected ')' or ']', got {self.peek()}")
            close = self.consume().value

            if len(elems) == 2:
                a, b = elems
                if close == ")":
                    return OpenInterval(a, b)
                else:
                    return OpenClosedInterval(a, b)

            return Point(None_(), elems)

        # |x|
        if tok.kind == "SYM" and tok.value == "|":
            self.consume()
            inner = self.parse_equation_chain()
            self.expect("SYM", "|")
            return Absolute(inner)

        # [ ... ] / [ ... )
        if tok.kind == "SYM" and tok.value == "[":
            open_idx = self.pos
            self.consume()
            end_idx, close_sym, _, _, has_comma_top, has_top_level_cmd_brace = self._scan_square_bracket_region()

            old_limit = self.limit
            start_pos = self.pos
            self.limit = end_idx
            prev_tok = self.tokens[open_idx - 1] if open_idx > 0 else None
            next_tok = self.tokens[end_idx + 1] if end_idx + 1 < len(self.tokens) else None
            wrap_gauss = True
            if has_top_level_cmd_brace and (
                _token_triggers_operator_context(prev_tok)
                or _token_triggers_operator_context(next_tok)
            ):
                wrap_gauss = False

            if has_comma_top:
                elems: List[Expression] = []
                elems.append(self.parse_equation_chain())
                while self.match("SYM", ",") and self.pos < self.limit:
                    self.consume()
                    elems.append(self.parse_equation_chain())

                if self.pos != self.limit:
                    self.pos = start_pos
                    self.limit = end_idx
                    inner = self.parse_equation_chain()
                    self.limit = old_limit
                    self.pos = end_idx + 1
                    return Gauss(inner) if wrap_gauss else inner

                self.limit = old_limit
                self.pos = end_idx + 1

                if len(elems) == 2:
                    a, b = elems
                    if close_sym == "]":
                        return ClosedInterval(a, b)
                    else:
                        return ClosedOpenInterval(a, b)
                else:
                    return Point(None_(), elems)

            inner = self.parse_equation_chain()
            self.limit = old_limit
            self.pos = end_idx + 1

            return Gauss(inner) if wrap_gauss else inner

        # plain { ... }
        if tok.kind == "SYM" and tok.value == "{":
            return self._parse_brace_block(as_group=False)

        # leading '_'
        if tok.kind == "SYM" and tok.value == "_":
            self.consume()
            sub = self._parse_group_or_atom()
            return Subscript(None_(), sub)

        # CMD
        if tok.kind == "CMD":
            cmd = tok.value
            self.consume()

            # 단위 파싱
            if cmd == r"\unit":
                # \unit{unit} 형식
                self.expect("SYM", "{")
                unit_expr = self.parse_unit_expression()
                self.expect("SYM", "}")
                return unit_expr

            if cmd == r"\mathrm":
                # \mathrm{unit} 형식
                self.expect("SYM", "{")
                unit_expr = self.parse_unit_expression()
                self.expect("SYM", "}")
                return unit_expr

            # \{ ... \}
            if cmd == r"\{":
                return self._parse_set_brace_block()

            if cmd == r"\}":
                raise ValueError("unexpected \\}")

            if cmd in (r"\int", r"\sum", r"\prod"):
                return self._parse_cmd_with_limits(cmd)

            # cases
            if cmd == r"\begin":
                self.expect("SYM", "{")
                env_tok = self.expect("ID")
                env_name = env_tok.value
                self.expect("SYM", "}")
                if env_name != "cases":
                    return Value(r"\begin{" + env_name + "}")

                rows: List[tuple[Expression, Expression]] = []

                while True:
                    tok2 = self.peek()
                    if tok2 is None:
                        break
                    if tok2.kind == "CMD" and tok2.value == r"\end":
                        break

                    row_end = self.limit
                    for i in range(self.pos, self.limit):
                        t = self.tokens[i]
                        if (t.kind == "CMD" and t.value in (r"\\", r"\end")) or \
                           (t.kind == "SYM" and t.value in {",", "&"}):
                            row_end = i
                            break

                    old_limit = self.limit
                    self.limit = row_end
                    row_expr = self.parse_equation_chain()
                    self.limit = old_limit
                    self.pos = row_end

                    cond_expr: Expression = None_()

                    if self.match("SYM", ","):
                        self.consume()

                    if self.match("SYM", "&") or self.match("CMD", r"\&"):
                        self.consume()
                        cond_end = self.limit
                        for i in range(self.pos, self.limit):
                            t = self.tokens[i]
                            if t.kind == "CMD" and t.value in (r"\\", r"\end"):
                                cond_end = i
                                break
                        old_limit = self.limit
                        self.limit = cond_end
                        cond_expr = self.parse_equation_chain()
                        self.limit = old_limit
                        self.pos = cond_end

                    if self.match("CMD", r"\\"):
                        self.consume()

                    cond_expr = _to_inline_phrase(cond_expr)
                    row_expr = _strip_endcases_expr(row_expr)
                    cond_expr = _strip_endcases_expr(cond_expr)

                    rows.append((row_expr, cond_expr))

                if self.match("CMD", r"\end"):
                    self.consume()
                    self.expect("SYM", "{")
                    end_name_tok = self.expect("ID")
                    self.expect("SYM", "}")

                return Cases(rows)

            # 상수/기호
            if cmd == r"\infty":
                return Infty()
            if cmd == r"\emptyset":
                return EmptySet()
            if cmd == r"\cdots":
                return Cdots()
            if cmd == r"\square":
                return Square()
            if cmd == r"\circ":
                return Degree()
            if cmd == r"\bigcirc":
                return BigCircle()
            if cmd == r"\uparrow":
                return UpArrow()
            if cmd == r"\downarrow":
                return DownArrow()
            if cmd == r"\leftarrow":
                return LeftArrow()
            if cmd == r"\rightarrow":
                return RightArrow()

            # 그리스 문자
            if cmd in self._GREEK_CMD_MAP:
                return Value(self._GREEK_CMD_MAP[cmd])

            # \text{...}
            if cmd == r"\text":
                open_tok = self.expect("SYM", "{")
                open_start = open_tok.start

                depth = 1
                i = self.pos
                close_idx = None
                while i < self.limit:
                    t = self.tokens[i]
                    if t.kind == "SYM" and t.value == "{":
                        depth += 1
                    elif t.kind == "SYM" and t.value == "}":
                        depth -= 1
                        if depth == 0:
                            close_idx = i
                            break
                    i += 1
                if close_idx is None:
                    raise ValueError("unclosed \\text{")

                close_tok = self.tokens[close_idx]
                raw = self.src[open_start + 1: close_tok.start]
                self.pos = close_idx + 1
                return Text(raw)

            # \mathrm{m}
            if cmd == r"\mathrm":
                arg = self._parse_group_or_atom()
                if isinstance(arg, Value):
                    return Unit(str(arg.val))
                return Unit(str(arg))

            # \mathbb{R}
            if cmd == r"\mathbb":
                arg = self._parse_group_or_atom()
                if isinstance(arg, Value) and str(arg.val) == "R":
                    return Value("ℝ")
                return Value(arg)

            # \vec{a}
            if cmd == r"\vec":
                arg = self._parse_group_or_atom()
                return Vec(arg)

            # \overline{AB}
            if cmd == r"\overline":
                if self.match("SYM", "{"):
                    self.consume()
                    if self.match("ID"):
                        name_tok = self.consume()
                        text = name_tok.value
                        self.expect("SYM", "}")
                        if len(text) >= 2:
                            return Segment(Value(text[0]), Value(text[1]))
                        return Value(r"\overline" + text)
                    else:
                        inner = self.parse_equation_chain()
                        self.expect("SYM", "}")
                        return inner
                return Value(cmd)

            # \angle
            if cmd == r"\angle":
                if self.match("ID"):
                    name_tok = self.consume()
                    return Angle(name_tok.value)
                if self.match("SYM", "{"):
                    self.consume()
                    if self.match("ID"):
                        name_tok = self.consume()
                        self.expect("SYM", "}")
                        return Angle(name_tok.value)
                    inner = self.parse_equation_chain()
                    self.expect("SYM", "}")
                    return Angle(str(inner))
                return Value(cmd)

            # \triangle
            if cmd == r"\triangle":
                if self.match("ID"):
                    name_tok = self.consume()
                    return Triangle(name_tok.value)
                if self.match("SYM", "{"):
                    self.consume()
                    if self.match("ID"):
                        name_tok = self.consume()
                        self.expect("SYM", "}")
                        return Triangle(name_tok.value)
                    inner = self.parse_equation_chain()
                    self.expect("SYM", "}")
                    return Triangle(str(inner))
                return Value(cmd)

            # \frac
            if cmd == r"\frac":
                num = self._parse_group_or_atom()
                denom = self._parse_group_or_atom()
                return Frac(num, denom)

            # \binom
            if cmd == r"\binom":
                upper = self._parse_group_or_atom()
                lower = self._parse_group_or_atom()
                return Comb(upper, lower)

            # \sqrt
            if cmd == r"\sqrt":
                index = Value(2)
                if self.match("SYM", "["):
                    self.consume()
                    idx_expr = self.parse_equation_chain()
                    self.expect("SYM", "]")
                    index = idx_expr
                rad = self._parse_group_or_atom()
                return SQRT(rad, index)

            # trig / ln
            if cmd in (r"\sin", r"\cos", r"\tan", r"\sec", r"\csc", r"\cot", r"\ln"):
                if self.match("SYM", "^"):
                    self.consume()
                    if self.match("SYM", "{"):
                        expo = self._parse_brace_block(as_group=True)
                    else:
                        if self.match("NUM"):
                            tok_num = self.consume()
                            expo = Value(int(tok_num.value))
                        elif self.match("ID"):
                            tok_id = self.consume()
                            name = tok_id.value
                            if len(name) == 1:
                                expo = Value(name)
                            else:
                                node: Expression = Value(name[0])
                                for ch in name[1:]:
                                    node = ImplicitMult(node, Value(ch))
                                expo = node
                        else:
                            expo = self.parse_atom()
                    arg = self._parse_function_arg()
                    if cmd == r"\sin":
                        trig = Sin(arg)
                    elif cmd == r"\cos":
                        trig = Cos(arg)
                    elif cmd == r"\tan":
                        trig = Tan(arg)
                    elif cmd == r"\sec":
                        trig = Sec(arg)
                    elif cmd == r"\csc":
                        trig = Csc(arg)
                    elif cmd == r"\cot":
                        trig = Cot(arg)
                    else:
                        trig = Ln(arg)
                    return Power(trig, expo)

                arg = self._parse_function_arg()
                if cmd == r"\sin":
                    return Sin(arg)
                if cmd == r"\cos":
                    return Cos(arg)
                if cmd == r"\tan":
                    return Tan(arg)
                if cmd == r"\sec":
                    return Sec(arg)
                if cmd == r"\csc":
                    return Csc(arg)
                if cmd == r"\cot":
                    return Cot(arg)
                if cmd == r"\ln":
                    return Ln(arg)

            # arcsin, arccos, arctan
            if cmd in (r"\arcsin", r"\arccos", r"\arctan"):
                arg = self._parse_function_arg()
                return Func(Value(cmd[1:]), [arg])

            # \log
            if cmd == r"\log":
                base = None_()
                if self.match("SYM", "_"):
                    self.consume()
                    if self.match("SYM", "{"):
                        base = self._parse_brace_block(as_group=True)
                    elif self.match("ID"):
                        name_tok = self.consume()
                        base = Value(name_tok.value)
                    elif self.match("NUM"):
                        num_tok = self.consume()
                        base = Value(int(num_tok.value))
                    else:
                        base = self.parse_atom()
                arg = self._parse_function_arg()
                return Log(base, arg)

            # \lim
            if cmd == r"\lim":
                var = None_()
                to = None_()

                self._consume_optional_limits()

                if self.match("SYM", "_"):
                    self.consume()
                    if self.match("SYM", "{"):
                        self.consume()
                        vtok = self.expect("ID")
                        var_name, inline_arrow = self._split_inline_arrow(vtok.value)
                        var = Value(var_name)
                        arrow_consumed = inline_arrow
                        if not arrow_consumed:
                            arrow_consumed = self._consume_to_like_arrow()
                        if not arrow_consumed:
                            raise ValueError("expected CMD \\to or \\rightarrow in limit subscript")
                        to = self._parse_limit_target()
                        self.expect("SYM", "}")
                    else:
                        vtok = self.expect("ID")
                        var_name, inline_arrow = self._split_inline_arrow(vtok.value)
                        var = Value(var_name)
                        arrow_consumed = inline_arrow
                        if not arrow_consumed:
                            arrow_consumed = self._consume_to_like_arrow()
                        if not arrow_consumed:
                            raise ValueError("expected CMD \\to or \\rightarrow in limit subscript")
                            to = self._parse_limit_target()

                expr = self.parse_add_sub()
                return Lim(var, to, expr)

            return Value(cmd)

        raise ValueError(f"unexpected token in atom: {tok}")

    def _consume_to_like_arrow(self) -> bool:
        tok = self.peek()
        if tok is None:
            return False

        arrow_cmds = {r"\to", r"\rightarrow", r"\leftarrow"}
        arrow_ids = {"arrow", "rightarrow", "leftarrow", "leftrightarrow"}

        if tok.kind == "CMD" and tok.value in arrow_cmds:
            self.consume()
            if self.match("ID") and self.peek().value in arrow_ids:
                self.consume()
            return True

        if tok.kind == "ID" and tok.value in arrow_ids:
            self.consume()
            return True

        if tok.kind == "SYM" and tok.value == "=":
            self.consume()
            return True

        return False

    def _consume_optional_limits(self):
        while True:
            tok = self.peek()
            if tok and tok.kind == "CMD" and tok.value in (r"\limits", r"\nolimits"):
                self.consume()
                continue
            break

    def _split_inline_arrow(self, name: str):
        suffixes = ("rightarrow", "leftarrow", "leftrightarrow", "arrow", "to")
        for suffix in suffixes:
            if name.endswith(suffix) and len(name) > len(suffix):
                base = name[: -len(suffix)]
                if base:
                    return base, True
        return name, False

    def _parse_limit_target(self) -> Expression:
        if self.match("CMD", r"\infty"):
            self.consume()
            return Infty()

        if self.match("SYM", "="):
            self.consume()

        if self.match("NUM"):
            num_tok = self.consume()
            base = Value(int(num_tok.value))
        elif self.match("ID"):
            base = Value(self.consume().value)
        else:
            base = self.parse_atom()

        tok = self.peek()
        if tok and tok.kind == "SYM" and tok.value in ("+", "-"):
            sign = tok.value
            self.consume()
            if sign == "+":
                return Add(base, None_())
            return Sub(base, None_())

        return base

    # =========================================================
    # \{ ... \} (집합)
    # =========================================================

    def _parse_set_brace_block(self) -> Expression:
        """
        CMD '\{' 를 consume 한 뒤 호출.

        - \{\}                 → EmptySet()
        - \{1,2,3\}            → SetRoster([...])
        - \{a_n\}              → Seq(Subscript(...))
        - \{x | cond\}, \{x:cond\} → SetBuilder(...)
        - **예외**:
          안쪽에 콤마/|/:가 없고,
          소괄호가 있고,
          산술 연산자(+,-,\times,\cdot,\div,/)가 있으면
          집합이 아니라 **연산 순서 괄호**로 본다 → 그냥 expr 반환.
        """
        open_idx = self.pos - 1
        end_idx, has_paren, has_arith, has_comma_top, has_bar_or_colon_top = self._scan_set_brace_end()
        force_set_literal = self._should_force_set_literal(open_idx, end_idx)

        old_limit = self.limit
        start_pos = self.pos
        self.limit = end_idx

        # 1) 괄호로 해석해야 하는 경우
        #    (집합이 아닌, 단순 연산 그룹)
        if has_paren and has_arith and not has_comma_top and not has_bar_or_colon_top and not force_set_literal:
            expr = self.parse_equation_chain()
            self.limit = old_limit
            self.pos = end_idx + 1   # '\}' 스킵
            return expr

        # 2) \{\}  → EmptySet
        if self.pos == end_idx:
            self.limit = old_limit
            self.pos = end_idx + 1
            return EmptySet()

        elems: List[Expression] = []

        # 첫 원소
        first = self.parse_equation_chain()
        elems.append(first)

        # 3) SetBuilder: \{ x | cond \}, \{ x : cond \}
        if self.match("SYM", "|") or self.match("SYM", ":"):
            self.consume()
            cond = self.parse_equation_chain()
            cond = _to_inline_phrase(cond)
            self.limit = old_limit
            self.pos = end_idx + 1
            return SetBuilder(first, cond)

        # 4) 콤마 기반 원소 나열
        while self.match("SYM", ","):
            self.consume()
            elems.append(self.parse_equation_chain())

        self.limit = old_limit
        self.pos = end_idx + 1  # '\}' 스킵

        # 5) 원소가 1개일 때: \{a_n\} → Seq, 그 외는 {x} 집합
        if len(elems) == 1:
            inner = elems[0]
            if isinstance(inner, Subscript):
                return Seq(inner)
            return SetRoster([inner])

        # 6) 일반 집합 {1,2,3,...}
        return SetRoster(elems)


    # =========================================================
    # plain { ... } 그룹
    # =========================================================

    def _parse_group_or_atom(self) -> Expression:
        if self.match("SYM", "{"):
            return self._parse_brace_block(as_group=True)
        return self.parse_atom()

    def _parse_brace_block(self, as_group: bool = False) -> Expression:
        self.expect("SYM", "{")

        if self.match("SYM", "}"):
            self.consume()
            return None_()

        elems: List[Expression] = []

        first = self.parse_equation_chain()
        elems.append(first)

        if self.match("SYM", "|") or self.match("SYM", ":"):
            self.consume()
            cond = self.parse_equation_chain()
            cond = _to_inline_phrase(cond)
            self.expect("SYM", "}")
            return SetBuilder(first, cond)

        while self.match("SYM", ","):
            self.consume()
            elem = self.parse_equation_chain()
            elems.append(elem)

        self.expect("SYM", "}")

        if len(elems) == 1:
            return elems[0]

        if as_group:
            return SetRoster(elems)

        return SetRoster(elems)

    # =========================================================
    # 함수 인자
    # =========================================================

    def _parse_function_arg(self) -> Expression:
        if self.match("SYM", "("):
            self.consume()
            expr = self.parse_equation_chain()
            self.expect("SYM", ")")
            return expr
        return self.parse_atom()

    def _parse_arg_list(self, allow_bar_separator: bool = False) -> List[Expression]:
        self.expect("SYM", "(")
        args: List[Expression] = []
        if self.match("SYM", ")"):
            self.consume()
            return args
        while True:
            arg = self.parse_equation_chain()
            args.append(arg)
            if self.match("SYM", ","):
                self.consume()
                continue
            if allow_bar_separator and (
                self.match("SYM", "|") or self.match("CMD", r"\mid")
            ):
                self.consume()
                continue
            if self.match("SYM", ")"):
                self.consume()
                break
            raise ValueError("expected ',' or ')' in arg list")
        return args

    def _parse_recurring_phrase(self) -> InlinePhrase:
        cmd_tok = self.consume()  # current token must be CMD
        if cmd_tok.value not in self._RECURRING_CMD_SET:
            raise ValueError(f"unexpected recurring marker {cmd_tok.value}")
        if self.match("SYM", "{"):
            recurring_expr = self._parse_brace_block(as_group=True)
        else:
            recurring_expr = self.parse_atom()
        return _ensure_inline_phrase(recurring_expr)


# =========================================================
# 후처리 (rewrite)
# =========================================================

def _flatten_implicit_raw(expr: Expression) -> List[Expression]:
    if isinstance(expr, ImplicitMult):
        return _flatten_implicit_raw(expr.left) + _flatten_implicit_raw(expr.right)
    return [expr]


def _rebuild_implicit_from_factors(factors: List[Expression]) -> Expression:
    if not factors:
        return None_()
    node = factors[0]
    for f in factors[1:]:
        node = ImplicitMult(node, f)
    return node


def _is_minus_one(expr: Expression) -> bool:
    if isinstance(expr, Value) and expr.val == -1:
        return True
    if isinstance(expr, Minus) and isinstance(expr.expr, Value) and expr.expr.val == 1:
        return True
    return False


def _is_integral_head(expr: Expression):
    if isinstance(expr, Value) and expr.val == r"\int":
        return True, None_(), None_()
    if isinstance(expr, Subscript) and isinstance(expr.base, Value) and expr.base.val == r"\int":
        lower = expr.sub
        return True, lower, None_()
    if (
        isinstance(expr, Power)
        and isinstance(expr.base, Subscript)
        and isinstance(expr.base.base, Value)
        and expr.base.base.val == r"\int"
    ):
        lower = expr.base.sub
        upper = expr.expo
        return True, lower, upper
    return False, None, None


def _is_sum_head(expr: Expression):
    if isinstance(expr, Value) and expr.val == r"\sum":
        return True, None_(), None_(), None_()
    if isinstance(expr, Subscript) and isinstance(expr.base, Value) and expr.base.val == r"\sum":
        sub = expr.sub
        var = None_()
        start = None_()
        if isinstance(sub, Eq):
            var = sub.left
            start = sub.right
        else:
            start = sub
        return True, var, start, None_()
    if (
        isinstance(expr, Power)
        and isinstance(expr.base, Subscript)
        and isinstance(expr.base.base, Value)
        and expr.base.base.val == r"\sum"
    ):
        ok, var, start, _ = _is_sum_head(expr.base)
        if not ok:
            return False, None, None, None
        end = expr.expo
        if isinstance(end, Seq):
            end = end.term
        return True, var, start, end
    return False, None, None, None


def _explode_inline_parts(parts: List[Expression]) -> List[Expression]:
    expanded: List[Expression] = []
    for part in parts:
        if (
            isinstance(part, Value)
            and isinstance(part.val, int)
            and abs(part.val) >= 10
        ):
            digits = str(abs(part.val))
            for d in digits:
                expanded.append(Value(int(d)))
        else:
            expanded.append(part)
    return expanded


def _expr_contains_text(expr: Expression) -> bool:
    if isinstance(expr, (Text, InlinePhrase)):
        return True
    if isinstance(expr, ImplicitMult):
        return any(_expr_contains_text(f) for f in _flatten_implicit_raw(expr))
    if isinstance(expr, Add):
        return _expr_contains_text(expr.left) or _expr_contains_text(expr.right)
    if isinstance(expr, Sub):
        return _expr_contains_text(expr.left) or _expr_contains_text(expr.right)
    return False


def _flatten_for_inline(expr: Expression) -> List[Expression]:
    if isinstance(expr, InlinePhrase):
        flattened: List[Expression] = []
        for part in expr.parts:
            flattened.extend(_flatten_for_inline(part))
        return flattened
    if isinstance(expr, Text):
        return [expr]
    if isinstance(expr, ImplicitMult):
        flattened: List[Expression] = []
        for factor in _flatten_implicit_raw(expr):
            flattened.extend(_flatten_for_inline(factor))
        return flattened
    if isinstance(expr, Add):
        return _flatten_for_inline(expr.left) + ["+"] + _flatten_for_inline(expr.right)
    if isinstance(expr, Sub):
        return _flatten_for_inline(expr.left) + ["-"] + _flatten_for_inline(expr.right)
    return [expr]


def _build_inline_phrase(expr: Expression) -> InlinePhrase:
    tokens = _flatten_for_inline(expr)
    parts: List[Expression] = []
    current_math: Optional[Expression] = None
    pending_op: Optional[str] = None

    def flush_current():
        nonlocal current_math
        if current_math is not None:
            parts.append(current_math)
            current_math = None

    for token in tokens:
        if isinstance(token, str):
            if token in ("+", "-"):
                pending_op = token
            continue

        if isinstance(token, Text):
            flush_current()
            parts.append(token)
            pending_op = None
            continue

        if current_math is None:
            current_math = token
        elif pending_op:
            if pending_op == "+":
                current_math = Add(current_math, token)
            else:
                current_math = Sub(current_math, token)
            pending_op = None
        else:
            current_math = ImplicitMult(current_math, token)

    flush_current()
    if not parts:
        parts.append(expr)
    return InlinePhrase(_explode_inline_parts(parts))


def _to_inline_phrase(expr: Expression) -> Expression:
    if isinstance(expr, None_):
        return expr
    if isinstance(expr, InlinePhrase):
        return InlinePhrase(_explode_inline_parts(expr.parts))
    if isinstance(expr, Text):
        return InlinePhrase([expr])

    if _expr_contains_text(expr):
        return _build_inline_phrase(expr)

    if isinstance(expr, ImplicitMult):
        factors = _flatten_implicit_raw(expr)
        return _rebuild_implicit_from_factors(factors)

    return expr


def _ensure_inline_phrase(expr: Expression) -> InlinePhrase:
    inline_candidate = _to_inline_phrase(expr)
    if isinstance(inline_candidate, InlinePhrase):
        return InlinePhrase(_explode_inline_parts(inline_candidate.parts))
    return InlinePhrase(_explode_inline_parts([inline_candidate]))


def _strip_endcases_expr(expr: Expression) -> Expression:
    if isinstance(expr, Eq):
        return Eq(expr.left, _strip_endcases_expr(expr.right))

    if isinstance(expr, ImplicitMult):
        factors = _flatten_implicit_raw(expr)
        cut_idx = None
        for i, f in enumerate(factors):
            if isinstance(f, Value) and getattr(f, "val", None) == r"\end":
                cut_idx = i
                break
        if cut_idx is not None:
            factors = factors[:cut_idx]
            if not factors:
                return None_()
            return _rebuild_implicit_from_factors(factors)
        return expr

    return expr


def _rewrite_implicit_chain(factors: List[Expression]) -> Optional[Expression]:
    n = len(factors)

    if n == 2 and isinstance(factors[0], Value) and factors[0].val == "P":
        event = factors[1]
        return Prob(event)

    if n >= 3 and isinstance(factors[0], Value) and isinstance(factors[-1], Expression):
        middle = factors[1:-1]
        if (
            len(middle) == 5
            and all(isinstance(x, Value) and isinstance(x.val, str) and len(x.val) == 1 for x in middle)
        ):
            mid_str = "".join(x.val for x in middle)
            if mid_str == "arrow":
                return Rimpl(factors[0], factors[-1])

    if n >= 4:
        d_part = factors[-2]
        var_part = factors[-1]
        if (
            isinstance(d_part, Value)
            and d_part.val == "d"
            and isinstance(var_part, Value)
            and isinstance(var_part.val, str)
            and len(var_part.val) == 1
        ):
            var_name = var_part.val
            for i in range(n - 3):
                head = factors[i]
                is_head, lower, upper = _is_integral_head(head)
                if not is_head:
                    continue
                inner_factors = factors[i + 1: n - 2]
                if not inner_factors:
                    integrand = Value(1)
                elif len(inner_factors) == 1:
                    integrand = inner_factors[0]
                else:
                    integrand = _rebuild_implicit_from_factors(inner_factors)
                lower_expr = lower if lower is not None else None_()
                upper_expr = upper if upper is not None else None_()
                integral_node = Integral(lower_expr, upper_expr, integrand, Value(var_name))
                prefix = factors[:i]
                new_factors = prefix + [integral_node]
                if not prefix:
                    return integral_node
                rewritten = _rewrite_implicit_chain(new_factors)
                if rewritten is not None:
                    return rewritten
                return _rebuild_implicit_from_factors(new_factors)

    if n >= 2:
        head = factors[0]
        is_head, var, start, end = _is_sum_head(head)
        if is_head:
            term_factors = factors[1:]
            if len(term_factors) == 1:
                term = term_factors[0]
            else:
                term = _rebuild_implicit_from_factors(term_factors)
            var_expr = var if var is not None else None_()
            start_expr = start if start is not None else None_()
            end_expr = end if end is not None else None_()
            return Sum(term, var_expr, start_expr, end_expr)

    if n == 2:
        left, right = factors
        if (
            isinstance(left, Subscript)
            and isinstance(left.base, None_)
            and isinstance(right, Subscript)
            and isinstance(right.base, Value)
            and isinstance(right.base.val, str)
        ):
            n_expr = left.sub
            r_expr = right.sub
            sym = right.base.val
            if sym == "P":
                return Perm(n_expr, r_expr)
            if sym == "C":
                return Comb(n_expr, r_expr)
            if sym == "Π":
                return RepeatedPermu(n_expr, r_expr)
            if sym == "H":
                return RepeatedComb(n_expr, r_expr)

    if n == 2:
        first, second = factors
        if isinstance(first, Power) and isinstance(first.base, Value):
            base = first.base
            expo = first.expo
            if _is_minus_one(expo):
                return FuncInv(base, [second])

    if n == 2 and isinstance(factors[0], Frac):
        frac = factors[0]
        arg_expr = factors[1]
        num = frac.num
        denom = frac.denom
        if isinstance(num, Value) and num.val == "d":
            if (
                isinstance(denom, ImplicitMult)
                and isinstance(denom.left, Value)
                and denom.left.val == "d"
                and isinstance(denom.right, Value)
            ):
                x_expr = denom.right
                inner = arg_expr
                return Diff(inner, x_expr, Value(1))

    return None


def _is_numeric_literal(val) -> bool:
    if isinstance(val, (int, float)):
        return True
    if isinstance(val, str):
        try:
            float(val)
            return True
        except ValueError:
            return False
    return False


def _is_numeric_coord(expr: Expression) -> bool:
    if isinstance(expr, Value) and _is_numeric_literal(expr.val):
        return True
    if isinstance(expr, RecurringDecimal):
        return True
    if isinstance(expr, Minus):
        return _is_numeric_coord(expr.expr)
    if isinstance(expr, Plus):
        return _is_numeric_coord(expr.expr)
    return False


def _is_prob_event_like(expr: Expression) -> bool:
    return not _is_numeric_coord(expr)


def _finalize_expr(expr: Expression, allow_interval: bool) -> Expression:
    if not allow_interval and isinstance(expr, OpenInterval):
        return Point(None_(), [expr.left, expr.right])
    return expr


def _is_integral_head_for_pow(expr: Expression) -> bool:
    if isinstance(expr, Value) and expr.val == r'\int':
        return True
    if isinstance(expr, Subscript) and isinstance(expr.base, Value) and expr.base.val == r'\int':
        return True
    if (
        isinstance(expr, Power)
        and isinstance(expr.base, Subscript)
        and isinstance(expr.base.base, Value)
        and expr.base.base.val == r'\int'
    ):
        return True
    return False


def _point_from_components(name_expr: Expression, coords: List[Expression], allow_interval: bool) -> Expression:
    is_unnamed = isinstance(name_expr, None_) or name_expr is None
    if is_unnamed and len(coords) == 2 and isinstance(coords[0], InnerProduct):
        return _finalize_expr(coords[0], allow_interval)
    return _finalize_expr(Point(name_expr, coords), allow_interval)


def _rewrite_expression(expr: Expression, allow_interval: bool = False) -> Expression:
    if isinstance(expr, (None_, Value, Infty, EulerNum, Cdots, EmptySet)):
        return expr

    if isinstance(expr, RecurringDecimal):
        non_recurring = _rewrite_expression(expr.non_recurring, allow_interval=True)
        recurring = _rewrite_expression(expr.recurring, allow_interval=True)
        return _finalize_expr(RecurringDecimal(non_recurring, recurring), allow_interval)

    if isinstance(expr, Add):
        return _finalize_expr(Add(_rewrite_expression(expr.left), _rewrite_expression(expr.right)), allow_interval)
    if isinstance(expr, Sub):
        return _finalize_expr(Sub(_rewrite_expression(expr.left), _rewrite_expression(expr.right)), allow_interval)
    if isinstance(expr, Mult):
        left = _rewrite_expression(expr.left)
        right = _rewrite_expression(expr.right)
        return _finalize_expr(Mult(left, right), allow_interval)
    if isinstance(expr, Divide):
        return _finalize_expr(Divide(_rewrite_expression(expr.left), _rewrite_expression(expr.right)), allow_interval)
    if isinstance(expr, Slash):
        return _finalize_expr(Slash(_rewrite_expression(expr.left), _rewrite_expression(expr.right)), allow_interval)
    if isinstance(expr, PlusMinus):
        return _finalize_expr(PlusMinus(_rewrite_expression(expr.left), _rewrite_expression(expr.right)), allow_interval)
    if isinstance(expr, MinusPlus):
        return _finalize_expr(MinusPlus(_rewrite_expression(expr.left), _rewrite_expression(expr.right)), allow_interval)

    if isinstance(expr, Eq):
        left = _rewrite_expression(expr.left)
        right = _rewrite_expression(expr.right)
        return _finalize_expr(Eq(left, right), allow_interval)

    if isinstance(expr, Less):
        return _finalize_expr(Less(_rewrite_expression(expr.left), _rewrite_expression(expr.right)), allow_interval)
    if isinstance(expr, Greater):
        return _finalize_expr(Greater(_rewrite_expression(expr.left), _rewrite_expression(expr.right)), allow_interval)
    if isinstance(expr, Leq):
        return _finalize_expr(Leq(_rewrite_expression(expr.left), _rewrite_expression(expr.right)), allow_interval)
    if isinstance(expr, Geq):
        return _finalize_expr(Geq(_rewrite_expression(expr.left), _rewrite_expression(expr.right)), allow_interval)
    if isinstance(expr, Neq):
        return _finalize_expr(Neq(_rewrite_expression(expr.left), _rewrite_expression(expr.right)), allow_interval)

    if isinstance(expr, SetCup):
        left = _rewrite_expression(expr.left, allow_interval=True)
        right = _rewrite_expression(expr.right, allow_interval=True)
        return _finalize_expr(SetCup(left, right), allow_interval)
    if isinstance(expr, SetCap):
        left = _rewrite_expression(expr.left, allow_interval=True)
        right = _rewrite_expression(expr.right, allow_interval=True)
        return _finalize_expr(SetCap(left, right), allow_interval)

    if isinstance(expr, SetIn):
        elem = _rewrite_expression(expr.elem)
        set_expr = _rewrite_expression(expr.set_expr, allow_interval=True)
        return _finalize_expr(SetIn(elem, set_expr), allow_interval)
    if isinstance(expr, SetNotIn):
        elem = _rewrite_expression(expr.elem)
        set_expr = _rewrite_expression(expr.set_expr, allow_interval=True)
        return _finalize_expr(SetNotIn(elem, set_expr), allow_interval)
    if isinstance(expr, SetSub):
        left = _rewrite_expression(expr.left, allow_interval=True)
        right = _rewrite_expression(expr.right, allow_interval=True)
        return _finalize_expr(SetSub(left, right), allow_interval)
    if isinstance(expr, SetSup):
        left = _rewrite_expression(expr.left, allow_interval=True)
        right = _rewrite_expression(expr.right, allow_interval=True)
        return _finalize_expr(SetSup(left, right), allow_interval)
    if isinstance(expr, SetNotSub):
        left = _rewrite_expression(expr.left, allow_interval=True)
        right = _rewrite_expression(expr.right, allow_interval=True)
        return _finalize_expr(SetNotSub(left, right), allow_interval)
    if isinstance(expr, SetNotSup):
        left = _rewrite_expression(expr.left, allow_interval=True)
        right = _rewrite_expression(expr.right, allow_interval=True)
        return _finalize_expr(SetNotSup(left, right), allow_interval)

    if isinstance(expr, Rimpl):
        return _finalize_expr(Rimpl(_rewrite_expression(expr.left), _rewrite_expression(expr.right)), allow_interval)
    if isinstance(expr, Limpl):
        return _finalize_expr(Limpl(_rewrite_expression(expr.left), _rewrite_expression(expr.right)), allow_interval)
    if isinstance(expr, Biconditional):
        return _finalize_expr(Biconditional(_rewrite_expression(expr.left), _rewrite_expression(expr.right)), allow_interval)
    if isinstance(expr, Iff):
        return _finalize_expr(Iff(_rewrite_expression(expr.left), _rewrite_expression(expr.right)), allow_interval)
    if isinstance(expr, Rsufficient):
        return _finalize_expr(Rsufficient(_rewrite_expression(expr.left), _rewrite_expression(expr.right)), allow_interval)
    if isinstance(expr, Lsufficient):
        return _finalize_expr(Lsufficient(_rewrite_expression(expr.left), _rewrite_expression(expr.right)), allow_interval)

    if isinstance(expr, Congru):
        return _finalize_expr(Congru(_rewrite_expression(expr.shape1), _rewrite_expression(expr.shape2)), allow_interval)
    if isinstance(expr, Sim):
        return _finalize_expr(Sim(_rewrite_expression(expr.shape1), _rewrite_expression(expr.shape2)), allow_interval)

    if isinstance(expr, Paral):
        l = _rewrite_expression(expr.line1)
        r = _rewrite_expression(expr.line2)
        if isinstance(l, Segment):
            l = Line(l.start, l.end)
        if isinstance(r, Segment):
            r = Line(r.start, r.end)
        return _finalize_expr(Paral(l, r), allow_interval)

    if isinstance(expr, Perp):
        l = _rewrite_expression(expr.line1)
        r = _rewrite_expression(expr.line2)
        if isinstance(l, Segment):
            l = Line(l.start, l.end)
        if isinstance(r, Segment):
            r = Line(r.start, r.end)
        return _finalize_expr(Perp(l, r), allow_interval)

    if isinstance(expr, Superscript):
        return _finalize_expr(Superscript(_rewrite_expression(expr.base), _rewrite_expression(expr.sup)), allow_interval)

    if isinstance(expr, Subscript):
        base_r = _rewrite_expression(expr.base)
        sub_r = _rewrite_expression(expr.sub)

        if (
            isinstance(base_r, Subscript)
            and isinstance(base_r.base, None_)
            and isinstance(base_r.sub, ImplicitMult)
        ):
            im = base_r.sub
            im_factors = _flatten_implicit_raw(im)
            if (
                len(im_factors) == 2
                and isinstance(im_factors[0], Value)
                and isinstance(im_factors[1], Value)
                and isinstance(sub_r, Value)
            ):
                n_val = im_factors[0]
                op_val = im_factors[1]
                if op_val.val == "P":
                    return Perm(n_val, sub_r)
                if op_val.val == "C":
                    return Comb(n_val, sub_r)
                if op_val.val == "Π":
                    return RepeatedPermu(n_val, sub_r)
                if op_val.val == "H":
                    return RepeatedComb(n_val, sub_r)

        if isinstance(base_r, Power) and isinstance(base_r.base, Gauss):
            antiderivative = base_r.base.expr
            upper = base_r.expo
            lower = sub_r
            return Integrated(antiderivative, lower, upper)

        return _finalize_expr(Subscript(base_r, sub_r), allow_interval)

    if isinstance(expr, Power):
        base_r = _rewrite_expression(expr.base)
        expo_r = _rewrite_expression(expr.expo)

        if isinstance(base_r, Subscript) and isinstance(base_r.base, Gauss):
            antiderivative = base_r.base.expr
            lower = base_r.sub
            upper = expo_r
            return Integrated(antiderivative, lower, upper)

        if isinstance(expo_r, Value) and str(expo_r.val) == "c":
            if not _is_integral_head_for_pow(base_r):
                if (
                    isinstance(base_r, Value)
                    and isinstance(base_r.val, str)
                    and len(base_r.val) == 1
                    and base_r.val.isupper()
                ):
                    return SetComple(base_r)
                if not (
                    isinstance(base_r, Value)
                    and isinstance(base_r.val, str)
                    and len(base_r.val) == 1
                    and base_r.val.islower()
                ):
                    return SetComple(base_r)

        return _finalize_expr(Power(base_r, expo_r), allow_interval)

    if isinstance(expr, Factorial):
        return _finalize_expr(Factorial(_rewrite_expression(expr.expr)), allow_interval)

    if isinstance(expr, SQRT):
        return _finalize_expr(SQRT(_rewrite_expression(expr.radicand), _rewrite_expression(expr.index)), allow_interval)

    if isinstance(expr, SetRoster):
        return _finalize_expr(SetRoster([_rewrite_expression(e, allow_interval=True) for e in expr.elements]), allow_interval)
    if isinstance(expr, Seq):
        return _finalize_expr(Seq(_rewrite_expression(expr.term, allow_interval=True)), allow_interval)

    if isinstance(expr, Sin):
        return _finalize_expr(Sin(_rewrite_expression(expr.arg)), allow_interval)
    if isinstance(expr, Cos):
        return _finalize_expr(Cos(_rewrite_expression(expr.arg)), allow_interval)
    if isinstance(expr, Tan):
        return _finalize_expr(Tan(_rewrite_expression(expr.arg)), allow_interval)
    if isinstance(expr, Sec):
        return _finalize_expr(Sec(_rewrite_expression(expr.arg)), allow_interval)
    if isinstance(expr, Csc):
        return _finalize_expr(Csc(_rewrite_expression(expr.arg)), allow_interval)
    if isinstance(expr, Cot):
        return _finalize_expr(Cot(_rewrite_expression(expr.arg)), allow_interval)
    if isinstance(expr, Ln):
        return _finalize_expr(Ln(_rewrite_expression(expr.arg)), allow_interval)
    if isinstance(expr, Log):
        return _finalize_expr(Log(_rewrite_expression(expr.base), _rewrite_expression(expr.arg)), allow_interval)
    if isinstance(expr, Lim):
        return _finalize_expr(
            Lim(
                _rewrite_expression(expr.var),
                _rewrite_expression(expr.to),
                _rewrite_expression(expr.expr),
            ),
            allow_interval,
        )

    if isinstance(expr, Func):
        name_r = _rewrite_expression(expr.name)
        args_r = [_rewrite_expression(a) for a in expr.args]

        if isinstance(name_r, Value) and name_r.val == "P":
            if args_r and all(_is_numeric_coord(a) for a in args_r):
                return _finalize_expr(Point(name_r, args_r), allow_interval)
            if len(args_r) == 1 and _is_prob_event_like(args_r[0]):
                return _finalize_expr(Prob(args_r[0]), allow_interval)
            if len(args_r) == 2 and _is_prob_event_like(args_r[0]):
                return _finalize_expr(Prob(args_r[0], args_r[1]), allow_interval)
            return _finalize_expr(Func(name_r, args_r), allow_interval)

        if (
            isinstance(name_r, Value)
            and isinstance(name_r.val, str)
            and len(name_r.val) == 1
            and name_r.val.isupper()
            and args_r
            and all(_is_numeric_coord(a) for a in args_r)
        ):
            return _finalize_expr(Point(name_r, args_r), allow_interval)

        return _finalize_expr(Func(name_r, args_r), allow_interval)
    if isinstance(expr, FuncInv):
        return _finalize_expr(
            FuncInv(
                _rewrite_expression(expr.name),
                [_rewrite_expression(a) for a in expr.args],
            ),
            allow_interval,
        )
    if isinstance(expr, Prime):
        return _finalize_expr(Prime(_rewrite_expression(expr.expr)), allow_interval)

    if isinstance(expr, Absolute):
        inner = _rewrite_expression(expr.expr)
        if isinstance(inner, OpenInterval):
            inner = Point(None_(), [inner.left, inner.right])
        if isinstance(inner, (Vec, Point)):
            return _finalize_expr(Norm(inner), allow_interval)
        return _finalize_expr(Absolute(inner), allow_interval)

    if isinstance(expr, Gauss):
        return _finalize_expr(Gauss(_rewrite_expression(expr.expr)), allow_interval)
    if isinstance(expr, Plus):
        return _finalize_expr(Plus(_rewrite_expression(expr.expr)), allow_interval)
    if isinstance(expr, Minus):
        return _finalize_expr(Minus(_rewrite_expression(expr.expr)), allow_interval)
    if isinstance(expr, Not):
        return _finalize_expr(Not(_rewrite_expression(expr.expr)), allow_interval)
    if isinstance(expr, Norm):
        return _finalize_expr(Norm(_rewrite_expression(expr.expr)), allow_interval)
    if isinstance(expr, Bar):
        return _finalize_expr(Bar(_rewrite_expression(expr.expr)), allow_interval)
    if isinstance(expr, SetComple):
        return _finalize_expr(SetComple(_rewrite_expression(expr.expr)), allow_interval)
    if isinstance(expr, Delta):
        return _finalize_expr(Delta(_rewrite_expression(expr.expr)), allow_interval)

    if isinstance(expr, InlinePhrase):
        parts = [_rewrite_expression(p, allow_interval=True) for p in expr.parts]
        return _finalize_expr(InlinePhrase(_explode_inline_parts(parts)), allow_interval)
    if isinstance(expr, Text):
        return expr

    if isinstance(expr, SetBuilder):
        var = _rewrite_expression(expr.var, allow_interval=True)
        cond = _rewrite_expression(expr.condition, allow_interval=True)
        return _finalize_expr(SetBuilder(var, cond), allow_interval)

    if isinstance(expr, Sum):
        return _finalize_expr(
            Sum(
                _rewrite_expression(expr.term),
                _rewrite_expression(expr.var),
                _rewrite_expression(expr.start),
                _rewrite_expression(expr.end),
            ),
            allow_interval,
        )

    if isinstance(expr, Integral):
        return _finalize_expr(
            Integral(
                _rewrite_expression(expr.lower, allow_interval=True),
                _rewrite_expression(expr.upper, allow_interval=True),
                _rewrite_expression(expr.integrand),
                _rewrite_expression(expr.var),
            ),
            allow_interval,
        )

    if isinstance(expr, Integrated):
        return _finalize_expr(
            Integrated(
                _rewrite_expression(expr.antiderivative),
                _rewrite_expression(expr.lower, allow_interval=True),
                _rewrite_expression(expr.upper, allow_interval=True),
            ),
            allow_interval,
        )

    if isinstance(expr, Frac):
        num_r = _rewrite_expression(expr.num)
        denom_r = _rewrite_expression(expr.denom)

        if isinstance(num_r, ImplicitMult):
            d_part = num_r.left
            y_expr = num_r.right

            n_expr = None
            if isinstance(d_part, Value) and d_part.val == "d":
                n_expr = Value(1)
            elif (
                isinstance(d_part, Power)
                and isinstance(d_part.base, Value)
                and d_part.base.val == "d"
            ):
                n_expr = d_part.expo

            if n_expr is not None:
                x_base = None

                if (
                    isinstance(denom_r, ImplicitMult)
                    and isinstance(denom_r.left, Value)
                    and denom_r.left.val == "d"
                ):
                    x_part = denom_r.right
                    if isinstance(x_part, Power):
                        x_base = x_part.base
                    else:
                        x_base = x_part

                elif (
                    isinstance(denom_r, Power)
                    and isinstance(denom_r.base, ImplicitMult)
                    and isinstance(denom_r.base.left, Value)
                    and denom_r.base.left.val == "d"
                ):
                    x_base = denom_r.base.right

                if x_base is not None:
                    return Diff(y_expr, x_base, n_expr)

        return _finalize_expr(Frac(num_r, denom_r), allow_interval)

    if isinstance(expr, ImplicitMult):
        raw_factors = _flatten_implicit_raw(expr)
        factors = [_rewrite_expression(f) for f in raw_factors]

        # 무한소수 특별 처리: Value(...) + Cdots() -> InlinePhrase([Value(...), Cdots()])
        # 예: 3.140\cdots -> InlinePhrase([Value(3.140), Cdots()])
        if any(isinstance(f, Cdots) for f in factors):
            return _finalize_expr(InlinePhrase(factors), allow_interval)

        # 온도 단위 특별 처리: Power(Value(number), Degree()) + Unit("C"/"F") -> Power(Value(number), Unit("℃"/"℉"))
        if (len(factors) == 2 and
            isinstance(factors[0], Power) and
            isinstance(factors[0].base, Value) and
            isinstance(factors[0].expo, Degree) and
            isinstance(factors[1], Unit) and
            factors[1].unit in ['C', 'F']):
            temp_unit = '℃' if factors[1].unit == 'C' else '℉'
            return _finalize_expr(Power(factors[0].base, Unit(temp_unit)), allow_interval)

        # 각도 단위 특별 처리: Power(Value(deg), Degree()) 뒤에 Prime(Value) 오는 경우
        # Prime(Value) -> Value + Unit("'") (분)
        # Prime(Prime(Value)) -> Value + Unit("''") (초)
        inline_parts = []
        has_angle_units = False
        has_degree = False  # Track if we've seen a Power(Value, Degree)
        i = 0
        while i < len(factors):
            f = factors[i]
            
            # Track Power(Value, Degree)
            if isinstance(f, Power) and isinstance(f.expo, Degree):
                has_degree = True
                inline_parts.append(f)
            # Check if current is Prime(Value) and we've seen a degree marker
            elif has_degree and isinstance(f, Prime):
                prime_inner = f.expr
                # Check for double prime (second): Prime(Prime(Value))
                if isinstance(prime_inner, Prime) and isinstance(prime_inner.expr, Value):
                    # This is seconds: ''
                    inline_parts.append(prime_inner.expr)
                    inline_parts.append(Unit("''"))
                    has_angle_units = True
                elif isinstance(prime_inner, Value):
                    # This is minutes: '
                    inline_parts.append(prime_inner)
                    inline_parts.append(Unit("'"))
                    has_angle_units = True
                else:
                    # Not angle notation
                    inline_parts.append(f)
            else:
                inline_parts.append(f)
            
            i += 1
        
        # If we found angle units, use InlinePhrase
        if has_angle_units:
            return _finalize_expr(InlinePhrase(inline_parts), allow_interval)
        
        replaced = _rewrite_implicit_chain(factors)
        if replaced is not None:
            return _finalize_expr(replaced, allow_interval)
        return _finalize_expr(_rebuild_implicit_from_factors(factors), allow_interval)

    if isinstance(expr, Point):
        name_src = expr.name if getattr(expr, "name", None) is not None else None_()
        name_r = _rewrite_expression(name_src) if not isinstance(name_src, None_) else name_src
        coords_src = getattr(expr, "args", getattr(expr, "coords", []))
        coords_r = [_rewrite_expression(c) for c in coords_src]
        return _point_from_components(name_r, coords_r, allow_interval)

    if isinstance(expr, Vec):
        return _finalize_expr(Vec(_rewrite_expression(expr.expr)), allow_interval)

    if isinstance(expr, InnerProduct):
        left = _rewrite_expression(getattr(expr, "vec1", getattr(expr, "left", expr)))
        right = _rewrite_expression(getattr(expr, "vec2", getattr(expr, "right", expr)))
        return _finalize_expr(InnerProduct(left, right), allow_interval)

    if isinstance(expr, ClosedInterval):
        return ClosedInterval(
            _rewrite_expression(expr.left, allow_interval=True),
            _rewrite_expression(expr.right, allow_interval=True),
        )

    if isinstance(expr, ClosedOpenInterval):
        return ClosedOpenInterval(
            _rewrite_expression(expr.left, allow_interval=True),
            _rewrite_expression(expr.right, allow_interval=True),
        )

    if isinstance(expr, OpenClosedInterval):
        return OpenClosedInterval(
            _rewrite_expression(expr.left, allow_interval=True),
            _rewrite_expression(expr.right, allow_interval=True),
        )

    if isinstance(expr, OpenInterval):
        left = _rewrite_expression(expr.left, allow_interval=True)
        right = _rewrite_expression(expr.right, allow_interval=True)
        if allow_interval:
            return OpenInterval(left, right)
        return _point_from_components(None_(), [left, right], allow_interval)

    if isinstance(expr, MixedFrac):
        return _finalize_expr(
            MixedFrac(
                _rewrite_expression(expr.whole),
                _rewrite_expression(expr.num),
                _rewrite_expression(expr.denom),
            ),
            allow_interval,
        )

    if isinstance(expr, Diff):
        return _finalize_expr(
            Diff(
                _rewrite_expression(expr.y),
                _rewrite_expression(expr.x),
                _rewrite_expression(expr.n),
            ),
            allow_interval,
        )

    if isinstance(expr, FuncDef):
        return _finalize_expr(
            FuncDef(_rewrite_expression(expr.func), _rewrite_expression(expr.mapping)),
            allow_interval,
        )

    if isinstance(expr, Prop):
        return _finalize_expr(
            Prop(_rewrite_expression(expr.symbol), _rewrite_expression(expr.statement)),
            allow_interval,
        )

    if isinstance(expr, Ratio):
        return _finalize_expr(
            Ratio([_rewrite_expression(r) for r in expr.ratios]),
            allow_interval,
        )

    if isinstance(expr, Segment):
        return _finalize_expr(
            Segment(_rewrite_expression(expr.start), _rewrite_expression(expr.end)),
            allow_interval,
        )

    if isinstance(expr, Ray):
        return _finalize_expr(
            Ray(_rewrite_expression(expr.start), _rewrite_expression(expr.through)),
            allow_interval,
        )

    if isinstance(expr, Line):
        return _finalize_expr(
            Line(_rewrite_expression(expr.point1), _rewrite_expression(expr.point2)),
            allow_interval,
        )

    if isinstance(expr, Perm):
        return _finalize_expr(Perm(_rewrite_expression(expr.n), _rewrite_expression(expr.r)), allow_interval)

    if isinstance(expr, Comb):
        return _finalize_expr(Comb(_rewrite_expression(expr.n), _rewrite_expression(expr.r)), allow_interval)

    if isinstance(expr, RepeatedPermu):
        return _finalize_expr(RepeatedPermu(_rewrite_expression(expr.n), _rewrite_expression(expr.r)), allow_interval)

    if isinstance(expr, RepeatedComb):
        return _finalize_expr(RepeatedComb(_rewrite_expression(expr.n), _rewrite_expression(expr.r)), allow_interval)

    if isinstance(expr, UnitDiv):
        return _finalize_expr(UnitDiv(_rewrite_expression(expr.num_unit), _rewrite_expression(expr.denom_unit)), allow_interval)

    if isinstance(expr, UnitMult):
        return _finalize_expr(UnitMult(_rewrite_expression(expr.left), _rewrite_expression(expr.right)), allow_interval)

    if isinstance(expr, About):
        return _finalize_expr(About(_rewrite_expression(expr.expr)), allow_interval)

    return expr


# =========================================================
# 공개 함수
# =========================================================

def latex_to_expression(latex: str) -> Expression:
    s = latex.strip()
    parser = LatexParser(s)
    expr = parser.parse_full()
    parser.expect_end()
    expr = _rewrite_expression(expr, allow_interval=True)
    return expr


# # LaTeX Parser 예시들

# In[8]:


latex_strings = [
    r"1+1=2< 3=1+2",
    r"[1, 2)",
    r"(1, 2]",
    r"[1, 2]",
    r"\frac{1}{2}",
    r"x^2 + 2x + 1",
    r"\vec{a}=(1, 2)",
    r"A(1, 3)",
    r"4\times [(1+3) \div {(1+3)\times 4}]",
    r"(1, 2, 3, 4, 5)",
    r"1+2+\cdots+10",
    r"[x]^1_3",
    r"\int^3_2 x dx",
    r"\lim_{h \rightarrow 1} h",
    r"\sum^3_{k=1} k",
    r"\int_5^4 adx + adx",
    r"\int^5_4 (adx + a)dx",
    r"AB \parallel CD",
    r"\{1, 2\}\cup \{1\} \cup \{x| x\text{는 } 10 \text{이하의 자연수} \}",
    r"1×2÷2",
    r"ㄱ",
]

expression_results = []
for latex_str in latex_strings:
    try:
        expr = latex_to_expression(latex_str)
        expression_results.append(expr)
        print(f"LaTeX: {latex_str}\nExpression: {str(expr)}\nRepr: {repr(expr)}\n")
    except Exception as e:
        print(f"Error processing '{latex_str}': {e}\n")

# Optionally, you can display the list of Expression objects
# display(expression_results)


# In[9]:


"""
LaTeX Parser - 100 Test Cases
제공된 LaTeX to Expression 변환 예제 100개에 대한 테스트
"""

# from expressions import *
# from latex_parser import latex_to_expression


# 테스트 케이스: (LaTeX, Expected Repr)
test_cases = [
    # Examples 1-10: Basic Operations
    (r"x", Value('x')),
    (r"2+3", Add(Value(2), Value(3))),
    (r"a-b", Sub(Value('a'), Value('b'))),
    (r"5 \times 7", Mult(Value(5), Value(7))),
    (r"\frac{1}{2}", Frac(Value(1), Value(2))),
    (r"x^2", Power(Value('x'), Value(2))),
    (r"-5", Minus(Value(5))),
    (r"|x|", Absolute(Value('x'))),
    (r"x = 5", Eq(Value('x'), Value(5))),
    (r"a < b", Less(Value('a'), Value('b'))),

    # Examples 11-20: Slightly More Complex
    (r"x^2 + 2x + 1", Add(Add(Power(Value('x'), Value(2)), ImplicitMult(Value(2), Value('x'))), Value(1))),
    (r"\frac{x+1}{x-1}", Frac(Add(Value('x'), Value(1)), Sub(Value('x'), Value(1)))),
    (r"2 \times (3+4)", Mult(Value(2), Add(Value(3), Value(4)))),
    (r"a^2 - b^2", Sub(Power(Value('a'), Value(2)), Power(Value('b'), Value(2)))),
    (r"\sqrt{x}", SQRT(Value('x'))),
    (r"x \leq 10", Leq(Value('x'), Value(10))),
    (r"3!", Factorial(Value(3))),
    (r"a \neq b", Neq(Value('a'), Value('b'))),
    (r"\frac{a}{b} \times \frac{c}{d}", Mult(Frac(Value('a'), Value('b')), Frac(Value('c'), Value('d')))),
    (r"(a+b)^2", Power(Add(Value('a'), Value('b')), Value(2))),

    # Examples 21-30: Intermediate Complexity
    (r"\frac{-b \pm \sqrt{b^2-4ac}}{2a}",
     Frac(PlusMinus(Minus(Value('b')), SQRT(Sub(Power(Value('b'), Value(2)), ImplicitMult(ImplicitMult(Value(4), Value('a')), Value('c'))))), ImplicitMult(Value(2), Value('a')))),
    (r"\sin(x) + \cos(x)", Add(Sin(Value('x')), Cos(Value('x')))),
    (r"\log_2{8}", Log(Value(2), Value(8))),
    (r"x_1 + x_2 + x_3", Add(Add(Subscript(Value('x'), Value(1)), Subscript(Value('x'), Value(2))), Subscript(Value('x'), Value(3)))),
    (r"\frac{d y}{d x}", Diff(Value('y'), Value('x'))),
    (r"\int x^2 dx", Integral(None_(), None_(), Power(Value('x'), Value(2)), Value('x'))),
    (r"\lim_{x \to 0} \frac{\sin x}{x}", Lim(Value('x'), Value(0), Frac(Sin(Value('x')), Value('x')))),
    (r"{}_5P_3", Perm(Value(5), Value(3))),
    (r"{}_5C_3", Comb(Value(5), Value(3))),
    (r"\{1, 2, 3, 4, 5\}", SetRoster([Value(1), Value(2), Value(3), Value(4), Value(5)])),

    # Examples 31-40: Sets and Logic
    (r"x \in A", SetIn(Value('x'), Value('A'))),
    (r"A \cup B", SetCup(Value('A'), Value('B'))),
    (r"A \cap B", SetCap(Value('A'), Value('B'))),
    (r"A \subset B", SetSub(Value('A'), Value('B'))),
    (r"A^c", SetComple(Value('A'))),
    (r"n(A)", SetNum(Value('A'))),
    (r"\{x | x < 5\}", SetBuilder(Value('x'), Less(Value('x'), Value(5)))),
    (r"p \rightarrow q", Rimpl(Value('p'), Value('q'))),
    (r"p \Rightarrow q", Rsufficient(Value('p'), Value('q'))),
    (r"\sim p", Not(Value('p'))),

    # Examples 41-50: Functions and Sequences
    (r"f(x) = x^2 + 1", Eq(Func(Value('f'), [Value('x')]), Add(Power(Value('x'), Value(2)), Value(1)))),
    (r"g(f(x))", Func(Value('g'), [Func(Value('f'), [Value('x')])])),
    (r"f^{-1}(x)", FuncInv(Value('f'), [Value('x')])),
    (r"\{a_n\}", Seq(Subscript(Value('a'), Value('n')))),
    (r"\sum_{k=1}^{n} k", Sum(Value('k'), Value('k'), Value(1), Value('n'))),
    (r"\sum_{k=1}^{\infty} \frac{1}{k^2}", Sum(Frac(Value(1), Power(Value('k'), Value(2))), Value('k'), Value(1), Infty())),
    (r"\lim_{n \to \infty} \frac{1}{n}", Lim(Value('n'), Infty(), Frac(Value(1), Value('n')))),
    (r"\int_0^1 x^2 dx", Integral(Value(0), Value(1), Power(Value('x'), Value(2)), Value('x'))),
    (r"[F(x)]_a^b", Integrated(Func(Value('F'), [Value('x')]), Value('a'), Value('b'))),
    (r"\frac{d^2 y}{dx^2}", Diff(Value('y'), Value('x'), Value(2))),

    # Examples 51-60: Trigonometry and Advanced Functions
    (r"\sin^2(x) + \cos^2(x) = 1", Eq(Add(Power(Sin(Value('x')), Value(2)), Power(Cos(Value('x')), Value(2))), Value(1))),
    (r"\tan(x) = \frac{\sin(x)}{\cos(x)}", Eq(Tan(Value('x')), Frac(Sin(Value('x')), Cos(Value('x'))))),
    (r"e^x", Power(EulerNum(), Value('x'))),
    (r"\ln(e^x) = x", Eq(Ln(Power(EulerNum(), Value('x'))), Value('x'))),
    (r"\log_a(xy) = \log_a(x) + \log_a(y)", Eq(Log(Value('a'), ImplicitMult(Value('x'), Value('y'))), Add(Log(Value('a'), Value('x')), Log(Value('a'), Value('y'))))),
    (r"\sqrt[3]{27}", SQRT(Value(27), Value(3))),
    (r"|x-5| < 3", Less(Absolute(Sub(Value('x'), Value(5))), Value(3))),
    (r"(n+1)!", Factorial(Add(Value('n'), Value(1)))),
    (r"\frac{n!}{(n-r)!r!}", Frac(Factorial(Value('n')), ImplicitMult(Factorial(Sub(Value('n'), Value('r'))), Factorial(Value('r'))))),
    (r"P(X=x)", Prob(Eq(Value('X'), Value('x')))),

    # Examples 61-70: Geometry and Vectors
    (r"\vec{a}", Vec(Value('a'))),
    (r"\vec{a} \cdot \vec{b}", InnerProduct(Vec(Value('a')), Vec(Value('b')))),
    (r"|\vec{a}|", Norm(Vec(Value('a')))),
    (r"\overline{AB}", Segment(Value('A'), Value('B'))),
    (r"\angle ABC", Angle('ABC')),
    (r"\triangle ABC", Triangle('ABC')),
    (r"\triangle ABC \equiv \triangle DEF", Congru(Triangle('ABC'), Triangle('DEF'))),
    (r"\triangle ABC \sim \triangle DEF", Sim(Triangle('ABC'), Triangle('DEF'))),
    (r"AB \parallel CD", Paral(Line(Value('A'), Value('B')), Line(Value('C'), Value('D')))),
    (r"AB \perp CD", Perp(Line(Value('A'), Value('B')), Line(Value('C'), Value('D')))),

    # Examples 71-80: Complex Expressions
    (r"\frac{a^2+b^2}{c^2+d^2} = \frac{x}{y}", Eq(Frac(Add(Power(Value('a'), Value(2)), Power(Value('b'), Value(2))), Add(Power(Value('c'), Value(2)), Power(Value('d'), Value(2)))), Frac(Value('x'), Value('y')))),
    (r"\sqrt{a^2 + b^2 + c^2}", SQRT(Add(Add(Power(Value('a'), Value(2)), Power(Value('b'), Value(2))), Power(Value('c'), Value(2))))),
    (r"\sin(2x) = 2\sin(x)\cos(x)", Eq(Sin(ImplicitMult(Value(2), Value('x'))), ImplicitMult(ImplicitMult(Value(2), Sin(Value('x'))), Cos(Value('x'))))),
    (r"\lim_{x \to \infty} (1 + \frac{1}{x})^x = e", Eq(Lim(Value('x'), Infty(), Power(Add(Value(1), Frac(Value(1), Value('x'))), Value('x'))), EulerNum())),
    (r"\int_a^b f(x) dx + \int_b^c f(x) dx = \int_a^c f(x) dx", Eq(Add(Integral(Value('a'), Value('b'), Func(Value('f'), [Value('x')]), Value('x')), Integral(Value('b'), Value('c'), Func(Value('f'), [Value('x')]), Value('x'))), Integral(Value('a'), Value('c'), Func(Value('f'), [Value('x')]), Value('x')))),
    (r"\sum_{i=1}^{n} (a_i + b_i) = \sum_{i=1}^{n} a_i + \sum_{i=1}^{n} b_i", Eq(Sum(Add(Subscript(Value('a'), Value('i')), Subscript(Value('b'), Value('i'))), Value('i'), Value(1), Value('n')), Add(Sum(Subscript(Value('a'), Value('i')), Value('i'), Value(1), Value('n')), Sum(Subscript(Value('b'), Value('i')), Value('i'), Value(1), Value('n'))))),
    (r"(A \cup B)^c = A^c \cap B^c", Eq(SetComple(SetCup(Value('A'), Value('B'))), SetCap(SetComple(Value('A')), SetComple(Value('B'))))),
    (r"_nP_r = \frac{n!}{(n-r)!}", Eq(Perm(Value('n'), Value('r')), Frac(Factorial(Value('n')), Factorial(Sub(Value('n'), Value('r')))))),
    (r"_nC_r = \frac{_nP_r}{r!}", Eq(Comb(Value('n'), Value('r')), Frac(Perm(Value('n'), Value('r')), Factorial(Value('r'))))),
    (r"P(A \cup B) = P(A) + P(B) - P(A \cap B)", Eq(Prob(SetCup(Value('A'), Value('B'))), Sub(Add(Prob(Value('A')), Prob(Value('B'))), Prob(SetCap(Value('A'), Value('B')))))),

    # Examples 81-90: Very Complex Expressions
    (r"\frac{d}{dx}(\sin(x^2 + 1)) = 2x\cos(x^2 + 1)", Eq(Diff(Sin(Add(Power(Value('x'), Value(2)), Value(1))), Value('x')), ImplicitMult(ImplicitMult(Value(2), Value('x')), Cos(Add(Power(Value('x'), Value(2)), Value(1)))))),
    (r"\int \frac{1}{x^2+1} dx = \arctan(x) + C", Eq(Integral(None_(), None_(), Frac(Value(1), Add(Power(Value('x'), Value(2)), Value(1))), Value('x')), Add(Func(Value('arctan'), [Value('x')]), Value('C')))),
    (r"\lim_{h \to 0} \frac{f(x+h) - f(x)}{h}", Lim(Value('h'), Value(0), Frac(Sub(Func(Value('f'), [Add(Value('x'), Value('h'))]), Func(Value('f'), [Value('x')])), Value('h')))),
    (r"\sum_{k=0}^{n} \binom{n}{k} x^k y^{n-k} = (x+y)^n", Eq(Sum(ImplicitMult(ImplicitMult(Comb(Value('n'), Value('k')), Power(Value('x'), Value('k'))), Power(Value('y'), Sub(Value('n'), Value('k')))), Value('k'), Value(0), Value('n')), Power(Add(Value('x'), Value('y')), Value('n')))),
    # (r"\prod_{i=1}^{n} (1 + a_i) = 1 + \sum_{i=1}^{n} a_i + \cdots", Eq(Func(Value('∏'), [Subscript(Value('i'), Value(1)), Value('n'), Add(Value(1), Subscript(Value('a'), Value('i')))]), Add(Add(Value(1), Sum(Subscript(Value('a'), Value('i')), Value('i'), Value(1), Value('n'))), Cdots()))),
    (r"\{x \in \mathbb{R} | x^2 - 5x + 6 = 0\}", SetBuilder(SetIn(Value('x'), Value('ℝ')), Eq(Add(Sub(Power(Value('x'), Value(2)), ImplicitMult(Value(5), Value('x'))), Value(6)), Value(0)))),
    (r"f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h}", Eq(Func(Prime(Value('f')), [Value('x')]), Lim(Value('h'), Value(0), Frac(Sub(Func(Value('f'), [Add(Value('x'), Value('h'))]), Func(Value('f'), [Value('x')])), Value('h'))))),
    (r"\int_0^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}", Eq(Integral(Value(0), Infty(), Power(EulerNum(), Minus(Power(Value('x'), Value(2)))), Value('x')), Frac(SQRT(Value('π')), Value(2)))),
    # (r"\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0", Eq(Add(Diff(Value('u'), Value('x'), Value(2)), Diff(Value('u'), Value('y'), Value(2))), Value(0))),
    (r"a=\begin{cases} x + y = 5 \\ x - y = 1 \end{cases}", Eq(Value("a"), Cases([(Eq(Add(Value('x'), Value('y')), Value(5)), None_()), (Eq(Sub(Value('x'), Value('y')), Value(1)), None_())]))),

    # Examples 91-100: Most Complex
    (r"\sum_{n=1}^{\infty} \frac{(-1)^{n+1}}{n} = \ln(2)", Eq(Sum(Frac(Power(Minus(Value(1)), Add(Value('n'), Value(1))), Value('n')), Value('n'), Value(1), Infty()), Ln(Value(2)))),
    (r"\lim_{n \to \infty} \sum_{k=1}^{n} \frac{k}{n^2} = \lim_{n \to \infty} \frac{n(n+1)}{2n^2}", Eq(Lim(Value('n'), Infty(), Sum(Frac(Value('k'), Power(Value('n'), Value(2))), Value('k'), Value(1), Value('n'))), Lim(Value('n'), Infty(), Frac(ImplicitMult(Value('n'), Add(Value('n'), Value(1))), ImplicitMult(Value(2), Power(Value('n'), Value(2))))))),
    (r"\int_0^{\pi} \sin^n(x) dx = \frac{n-1}{n} \int_0^{\pi} \sin^{n-2}(x) dx", Eq(Integral(Value(0), Value('π'), Power(Sin(Value('x')), Value('n')), Value('x')), ImplicitMult(Frac(Sub(Value('n'), Value(1)), Value('n')), Integral(Value(0), Value('π'), Power(Sin(Value('x')), Sub(Value('n'), Value(2))), Value('x'))))),
    (r"\frac{d}{dx}\left(\int_a^x f(t) dt\right) = f(x)", Eq(Diff(Integral(Value('a'), Value('x'), Func(Value('f'), [Value('t')]), Value('t')), Value('x')), Func(Value('f'), [Value('x')]))),
    (r"e^{i\pi} + 1 = 0", Eq(Add(Power(EulerNum(), ImplicitMult(Value('i'), Value('π'))), Value(1)), Value(0))),
    # (r"\left|\begin{matrix} a & b \\ c & d \end{matrix}\right| = ad - bc", Eq(Func(Value('det'), [Func(Value('matrix'), [Value('a'), Value('b'), Value('c'), Value('d')])]), Sub(ImplicitMult(Value('a'), Value('d')), ImplicitMult(Value('b'), Value('c'))))),
    (r"\sum_{k=0}^{n} (-1)^k \binom{n}{k} = 0", Eq(Sum(ImplicitMult(Power(Minus(Value(1)), Value('k')), Comb(Value('n'), Value('k'))), Value('k'), Value(0), Value('n')), Value(0))),
    (r"\lim_{x \to 0+} x^x = 1", Eq(Lim(Value('x'), Add(Value(0), None_()), Power(Value('x'), Value('x'))), Value(1))),
    (r"\int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} dx = 1", Eq(Integral(Minus(Infty()), Infty(), ImplicitMult(Frac(Value(1), SQRT(ImplicitMult(ImplicitMult(Value(2), Value('π')), Power(Value('σ'), Value(2))))), Power(EulerNum(), Minus(Frac(Power(Sub(Value('x'), Value('μ')), Value(2)), ImplicitMult(Value(2), Power(Value('σ'), Value(2))))))), Value('x')), Value(1))),
    (r"""    F(m,n)=
        \begin{cases}
          f(n),& \text{for } (0\leq n \leq 1) \\
          f(n-1),& (1< n \leq 2) \\
          f(n-1),&
          f(3)\\
          f(4)
        \end{cases}""",
      Eq( Func(Value("F"), [Value("m"), Value("n")]),
      Cases( [
      (Func(Value("f"), [Value("n")]), InlinePhrase([Text("for "), Leq(Value(0), Leq(Value("n"), Value(1))) ])),
      (Func(Value("f"), [Sub(Value("n"), Value(1))]), Less(Value(1), Leq(Value("n"), Value(2))) ),
      (Func(Value("f"), [Sub(Value("n"), Value(1))]), Func(Value("f"), [Value(3)]) ),
      (Func(Value("f"), [Value(4)]), None_() ),
      ]))
    ),
    (r"(1, 2) \cdot (2, 4)", InnerProduct(Point(None_(), [Value(1), Value(2)]), Point(None_(), [Value(2), Value(4)]))),
    (r"1.1+0.0\dot{9}+0.\dot{1}2ab3\dot{5}>12.345", Greater(Add(Add(Value("1.1"), RecurringDecimal(Value("0.0"), InlinePhrase([Value("9")]))), RecurringDecimal(Value("0"), InlinePhrase([Value("1"), Value("2"), Value("a"), Value("b"), Value("3"), Value("5")]))), Value(12.345)) ),
    (r"\sum\limits_{x=1}^{3} x", Sum(Value("x"), Value("x"), Value(1), Value(3) ) ),
    (r"_n\Pi_r", RepeatedPermu(Value("n"), Value("r"))),
    (r"_nH_r", RepeatedComb(Value("n"), Value("r"))),
    (r"\lim\limits_{n\rightarrow 1+} n^3+n^4", Lim(Value("n"), Add(Value(1), None_()), Add(Power(Value("n"), Value(3)),  Power(Value("n"), Value(4))) )),
    (r"1+1=2< 3=1+2", Eq(Add(Value(1), Value(1)), Less(Value(2), Eq(Value(3), Add(Value(1), Value(2)))))),
    (r"(1, 2)", OpenInterval(Value(1), Value(2))),
    (r"[1, 2)",ClosedOpenInterval(Value(1), Value(2)) ),
    (r"(1, 2]",OpenClosedInterval(Value(1), Value(2)) ),
    (r"[1, 2]", ClosedInterval(Value(1),Value(2)) ),
    (r"\vec{a}=((1, 2) \cdot (2, 4), 7) = (10, 7)", Eq(Vec(Value("a")), Eq(InnerProduct(Point(None_(), [Value(1), Value(2)]),Point(None_(), [Value(2), Value(4)]) ), Point(None_(), [Value(10), Value(7)])   ))  ),
    (r"A(1, 3)", Point(Value("A"), [Value(1), Value(3)]) ),
    (r"4\times [(1+3) \div {(1+3)\times 4}]", Mult(Value(4), Gauss(Divide(Add(Value(1), Value(3)), Mult(Add(Value(1), Value(3)), Value(4)))))),
    (r"4\times [(1+3) \div \{(1+3)\times 4\}]", Mult(Value(4), Divide(Add(Value(1), Value(3)), Mult(Add(Value(1), Value(3)), Value(4))))),
    (r"4\times [(1+3) \div (n(\{1+3\})\times 4)]", Mult(Value(4), Gauss(Divide(Add(Value(1), Value(3)), Mult(  SetNum(SetRoster([Add(Value(1), Value(3))])), Value(4)))))),
    (r"[(1+3) \div \{(1+3)\times 4\}]", Gauss(Divide(Add(Value(1), Value(3)), Mult(Add(Value(1), Value(3)), Value(4)))) ),
    (r"\{(1+3)\times 4\}", SetRoster([Mult(Add(Value(1), Value(3)), Value(4))]) ),
    (r"\{(1+3)\times 4\}\div A", Divide(Mult(Add(Value(1), Value(3)), Value(4)), Value("A") ) ),
    (r"\{(1+3)\times 4\}\cap A", SetCap(SetRoster([Mult(Add(Value(1), Value(3)), Value(4))]), Value("A"))),
    (r"(1, 2, 3, 4, 5)", Point(None, [Value(1), Value(2), Value(3), Value(4), Value(5)])),
    (r"1+2+\cdots+10", Add(Add(Add(Value(1), Value(2)), Cdots()), Value(10)) ),
    (r"[x]^1_3", Integrated(Value("x"), Value(3), Value(1))),
    (r"\int_5^4 adx + adx", Add(Integral(Value(5), Value(4), Value("a"), Value("x")), ImplicitMult(ImplicitMult(Value("a"), Value("d")), Value("x")))),
    (r"\int^5_4 (adx + a)dx", Integral(Value(4), Value(5), Add(ImplicitMult(ImplicitMult(Value("a"), Value("d")), Value("x")), Value("a")), Value("x"))),
    (r"AB \parallel CD", Paral(Line(Value("A"), Value("B")), Line(Value("C"), Value("D")))),
    (r"\{1, 2\}\cup \{1\} \cup \{x| x\text{는 } 10 \text{이하의 자연수} \}", SetCup(SetCup(SetRoster([Value(1), Value(2)]), SetRoster([Value(1)])), SetBuilder(Value("x"), InlinePhrase([Value("x"), Text("는 "), Value(1), Value(0), Text("이하의 자연수")])))),
    (r"(1, 10.\dot{1}] \cap (-1, 5) \cup [0, 6] \cap [-0.5, 6)",
     SetCap(
        SetCup(
            SetCap(
                OpenClosedInterval(Value(1), RecurringDecimal(Value("10"), InlinePhrase([Value("1")]))),
                OpenInterval(Minus(Value(1)), Value(5))),
            ClosedInterval(Value(0), Value(6))),
        ClosedOpenInterval(Minus(Value("0.5")), Value(6))   )  ),

    # Unit Test Cases 1-30: 초등학교~고등학교 수준 단위 포함 수식
    (r"5 \mathrm{kg}", ImplicitMult(Value(5), Unit("kg"))),
    (r"\unit{g}", Unit("g")),
    (r"2.5 \mathrm{mg}", ImplicitMult(Value("2.5"), Unit("mg"))),
    (r"\mathrm{\mu g}", UnitMult(Unit("mu"), Unit("g"))),
    (r"10 \mathrm{m}", ImplicitMult(Value(10), Unit("m"))),
    (r"3.14 \mathrm{cm}", ImplicitMult(Value("3.14"), Unit("cm"))),
    (r"1000 \mathrm{mm}", ImplicitMult(Value(1000), Unit("mm"))),
    (r"1.5 \mathrm{km}", ImplicitMult(Value("1.5"), Unit("km"))),
    (r"60 \mathrm{s}", ImplicitMult(Value(60), Unit("s"))),
    (r"5 \mathrm{min}", ImplicitMult(Value(5), Unit("min"))),
    (r"2 \mathrm{h}", ImplicitMult(Value(2), Unit("h"))),
    (r"25^\circ \mathrm{C}", Power(Value(25), Unit("℃"))),
    (r"77^\circ \mathrm{F}", Power(Value(77), Unit("℉"))),
    (r"300 \mathrm{K}", ImplicitMult(Value(300), Unit("K"))),
    (r"1 \mathrm{mol}", ImplicitMult(Value(1), Unit("mol"))),
    (r"0.5 \mathrm{mmol}", ImplicitMult(Value("0.5"), Unit("mmol"))),
    (r"\mathrm{pH} = 7", Eq(Unit("pH"), Value(7))),
    (r"90^\circ", Power(Value(90), Degree())),
    (r"\text{북위 } 35^\circ 30'\ \ 5.5''", InlinePhrase([Text("북위 "), Power(Value(35), Degree()), Value(30), Unit("'"), Value(5.5), Unit("''") ])  ),
    (r"\text{북위 } 35^\circ\ 30' \ 5.5''", InlinePhrase([Text("북위 "), Power(Value(35), Degree()), Value(30), Unit("'"), Value(5.5), Unit("''") ])  ),    
    (r"1 \mathrm{pc}", ImplicitMult(Value(1), Unit("pc"))),
    (r"4.37 \mathrm{ly}", ImplicitMult(Value("4.37"), Unit("ly"))),
    (r"1 \mathrm{nm}^3", ImplicitMult(Value(1), Power(Unit("nm"), Value(3)))),
    (r"\mathrm{kg} \cdot \mathrm{m}^2 / \mathrm{s}^2", Slash(Mult(Unit("kg"), Power(Unit("m"), Value(2))), Power(Unit("s"), Value(2)))),
    (r"9.8 \mathrm{m/s^2}", ImplicitMult(Value("9.8"), UnitDiv(Unit("m"), Power(Unit("s"), Value(2))))),
    (r"3.00 \times 10^8 \mathrm{m/s}", Mult(Value("3.00"), ImplicitMult(Power(Value(10), Value(8)), UnitDiv(Unit("m"), Unit("s"))))),
    (r"6.02 \times 10^{23} \mathrm{mol^{-1}}", Mult(Value("6.02"), ImplicitMult(Power(Value(10), Value(23)), Power(Unit("mol"), Minus(Value(1)))))),
    (r"22.4 \mathrm{L/mol}", ImplicitMult(Value("22.4"), UnitDiv(Unit("L"), Unit("mol")))),
    (r"1 \mathrm{atm}", ImplicitMult(Value(1), Unit("atm"))),
    (r"760 \mathrm{mmHg}", ImplicitMult(Value(760), Unit("mmHg"))),
    (r"101325 \mathrm{Pa}", ImplicitMult(Value(101325), Unit("Pa"))),
    (r"1.602 \times 10^{-19} \mathrm{C}", Mult(Value("1.602"), ImplicitMult(Power(Value(10), Minus(Value(19))), Unit("C")))),
    (r"6.626 \times 10^{-34} \mathrm{J \cdot s}", Mult(Value("6.626"), ImplicitMult(Power(Value(10), Minus(Value(34))), UnitMult(Unit("J"), Unit("s"))))),
    (r"\angle A +\angle B=45^\circ+60^\circ=105^\circ", Eq(Add(Angle(Value("A")), Angle(Value("B")) )  , Eq(Add(Power(Value(45), Degree()), Power(Value(60), Degree())) , Power(Value(105), Degree()))   )),

    # Ratio Test Cases
    (r"1:2", Ratio([Value(1), Value(2)])),
    (r"1:2:3", Ratio([Value(1), Value(2), Value(3)])),
    (r"a:b:c:d", Ratio([Value("a"), Value("b"), Value("c"), Value("d")])),
    (r"1:2=3:x", Eq(Ratio([Value(1), Value(2)]), Ratio([Value(3), Value("x")]))),
    (r"2(-5x+1):4=-5:8(3x+4)", Eq(Ratio([Func(Value(2), [Add(ImplicitMult(Minus(Value(5)), Value("x")), Value(1))]), Value(4)]), Ratio([Minus(Value(5)), Func(Value(8), [Add(ImplicitMult(Value(3), Value("x")), Value(4))])]))),

    (r" \bigcirc +\square =3", Eq( Add(BigCircle(), Square()) , Value(3)) ),
    (r"\pi \fallingdotseq 3.141592 \approx 3", Eq( Value("π"), About(Eq( Value(3.141592), About(Value(3))  )) ) ),
    (r"\frac{d}{dx} [f(x)]", Diff(Gauss(Func(Value("f"), [Value("x")])), Value("x"), Value(1)) ),
]

def normalize_repr(s: str) -> str:
    """Repr 문자열 정규화 (공백, 따옴표 처리)"""
    # 공백 제거
    s = s.replace(" ", "")
    # 작은따옴표와 큰따옴표 통일
    s = s.replace('"', "'")
    return s


def run_tests():
    """100개 테스트 실행"""
    print("=" * 80)
    print("LaTeX Parser - 100 Test Cases")
    print("=" * 80)
    print()

    passed = 0
    failed = 0
    errors = []

    for i, (latex, expected) in enumerate(test_cases, 1):
        try:
            expr = latex_to_expression(latex)
            result = repr(expr)

            # 정규화하여 비교
            result_norm = str(expr)
            expected_norm = str(expected)

            if result_norm == expected_norm:
                print(f"✓ Test {i:3d}: PASS")
                print(f"{latex}")
                passed += 1
            else:
                print(f"✗ Test {i:3d}: FAIL")
                print(f"  LaTeX:    {latex}")
                print(f"  Expected: {repr(expected)}")
                print(f"  Got:      {result}")
                failed += 1
                errors.append((i, latex, expected, result))
        except Exception as e:
            print(f"✗ Test {i:3d}: ERROR")
            print(f"  LaTeX: {latex}")
            print(f"  Error: {str(e)}")
            failed += 1
            errors.append((i, latex, expected, str(e)))

    print()
    print("=" * 80)
    print(f"Summary: {passed} PASSED, {failed} FAILED out of {len(test_cases)} tests")
    print(f"Success Rate: {passed / len(test_cases) * 100:.1f}%")
    print("=" * 80)

    if errors:
        print()
        print("Failed Tests Details:")
        print("-" * 80)
        for i, latex, expected, got in errors:
            print(f"\nTest {i}:")
            print(f"  LaTeX:    {latex}")
            print(f"  Expected: {repr(expected)}")
            print(f"  Got:      {got}")

        if len(errors) > 10:
            print(f"\n... and {len(errors) - 10} more failures")

    return passed, failed, errors


def run_single_test(test_num: int):
    """특정 테스트만 실행"""
    if test_num < 1 or test_num > len(test_cases):
        print(f"Invalid test number. Must be between 1 and {len(test_cases)}")
        return

    latex, expected = test_cases[test_num - 1]

    print(f"Test {test_num}:")
    print(f"LaTeX:    {latex}")
    print(f"Expected: {expected}")
    print()

    try:
        expr = latex_to_expression(latex)
        result = repr(expr)

        print(f"Result:   {result}")
        print()

        result_norm = str(result)
        expected_norm = str(expected)

        if result_norm == expected_norm:
            print("✓ PASS")
        else:
            print("✗ FAIL")
            print()
            print("Normalized comparison:")
            print(f"Expected: {expected_norm}")
            print(f"Got:      {result_norm}")
    except Exception as e:
        print(f"✗ ERROR: {e}")


def run_range_tests(start: int, end: int):
    """범위의 테스트 실행"""
    print(f"Running tests {start} to {end}")
    print("=" * 80)

    passed = 0
    failed = 0

    for i in range(start, end + 1):
        if i < 1 or i > len(test_cases):
            continue

        latex, expected = test_cases[i - 1]

        try:
            expr = latex_to_expression(latex)
            result = repr(expr)
            print(expr) 
            print(result)

            result_norm = result
            expected_norm = repr(expr)

            if result_norm == expected_norm:
                print(f"✓ Test {i}: PASS")
                passed += 1
            else:
                print(f"✗ Test {i}: FAIL - {latex}")
                failed += 1
        except Exception as e:
            print(f"✗ Test {i}: ERROR - {str(e)}")
            failed += 1

    print("=" * 80)
    print(f"Range {start}-{end}: {passed} PASSED, {failed} FAILED")
