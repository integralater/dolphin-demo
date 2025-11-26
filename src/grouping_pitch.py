# Expression 객체를 입력받아 [(token, pitch_level, volume_level), ...] 반환

from typing import List, Tuple
from LaTeX_Parser import *
from speech_synthesizer import *
from Expression_Syntax import *
from gtts_expr_audio_pitch import *
from audio_pitch import *


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
        Not, Rimpl, Limpl, Prop,
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
        Rimpl, Limpl,
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

def expr_to_tokens_with_pitch(expr, d=0) -> List[Tuple[str, int, int]]:
    """
    Expression을 [(token, pitch_level, volume_level), ...] 로 변환

    Args:
        expr: Expression 객체
        d: 현재 pitch level (기본값 0)

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

        a_tokens = expr_to_tokens_with_pitch(expr.non_recurring, d)
        b_tokens = expr_to_tokens_with_pitch(expr.recurring, d)

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
        return [("동그라미", d, 0)]

    if isinstance(expr, EulerNum):
        return [("자연상수", d, 0), ("이", d, 0)]

    # ==================== 단항 연산자 ====================

    if isinstance(expr, Absolute):
        x = expr.expr
        x_d = d + 1 if is_mid_or_postfix(x) else d
        tokens.append(("절댓값", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(x, x_d))
        return tokens

    if isinstance(expr, Gauss):
        x = expr.expr
        x_d = d + 1 if is_mid_or_postfix(x) else d
        tokens.append(("가우스", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(x, x_d))
        return tokens

    if isinstance(expr, Plus):
        a = expr.expr
        a_d = d + 1 if is_mid_or_postfix(a) else d
        tokens.append(("플러스", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(a, a_d))
        return tokens

    if isinstance(expr, Minus):
        a = expr.expr
        a_d = d + 1 if is_mid_or_postfix(a) else d
        tokens.append(("마이너스", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(a, a_d))
        return tokens

    if isinstance(expr, Not):
        p = expr.expr
        p_d = d + 1 if is_mid_or_postfix(p) else d
        tokens.append(("낫", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(p, p_d))
        return tokens

    if isinstance(expr, Delta):
        x = expr.expr
        x_d = d + 1 if is_mid_or_postfix(x) else d
        tokens.append(("델타", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(x, x_d))
        return tokens

    if isinstance(expr, Vec):
        a = expr.expr
        tokens.append(("벡터", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(a, d))
        return tokens

    if isinstance(expr, Norm):
        v = expr.expr
        v_d = d + 1 if is_mid_or_postfix(v) else d
        tokens.append(("노름", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(v, v_d))
        return tokens

    # ==================== 후위 연산자 ====================

    if isinstance(expr, Factorial):
        n = expr.expr
        n_d = d + 1 if is_operator(n) else d
        tokens.extend(expr_to_tokens_with_pitch(n, n_d))
        tokens.append(("팩토리얼", d, 0))
        return tokens

    if isinstance(expr, Prime):
        y = expr.expr
        # y가 연산자이고 Prime이 아니면 d+1
        if is_operator(y) and not isinstance(y, Prime):
            y_d = d + 1
        else:
            y_d = d
        tokens.extend(expr_to_tokens_with_pitch(y, y_d))
        tokens.append(("프라임", d, 0))
        return tokens

    if isinstance(expr, SetComple):
        A = expr.expr
        A_d = d + 1 if is_operator(A) else d
        tokens.extend(expr_to_tokens_with_pitch(A, A_d))
        tokens.append(("의", d, 0))
        tokens.append(("여집합", d, 0))
        return tokens

    if isinstance(expr, Bar):
        X = expr.expr
        X_d = d + 1 if is_operator(X) else d
        tokens.extend(expr_to_tokens_with_pitch(X, X_d))
        tokens.append(("바", d, 0))
        return tokens

    # ==================== 2항 연산자: 덧셈/뺄셈 ====================

    if isinstance(expr, Add):
        a, b = expr.left, expr.right
        # None_인지 확인 (우극한/좌극한)
        if isinstance(b, None_):
            tokens.extend(expr_to_tokens_with_pitch(a, d))
            tokens.append(("플러스", d, 0))
            return tokens

        tokens.extend(expr_to_tokens_with_pitch(a, d))
        b_d = d + 1 if is_add_like(b) else d
        tokens.append(("더하기", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(b, b_d))
        return tokens

    if isinstance(expr, Sub):
        a, b = expr.left, expr.right
        # None_인지 확인
        if isinstance(b, None_):
            tokens.extend(expr_to_tokens_with_pitch(a, d))
            tokens.append(("마이너스", d, 0))
            return tokens

        tokens.extend(expr_to_tokens_with_pitch(a, d))
        b_d = d + 1 if is_add_like(b) else d
        tokens.append(("빼기", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(b, b_d))
        return tokens

    if isinstance(expr, PlusMinus):
        a, b = expr.left, expr.right
        tokens.extend(expr_to_tokens_with_pitch(a, d))
        b_d = d + 1 if is_add_like(b) else d
        tokens.append(("플마", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(b, b_d))
        return tokens

    if isinstance(expr, MinusPlus):
        a, b = expr.left, expr.right
        tokens.extend(expr_to_tokens_with_pitch(a, d))
        b_d = d + 1 if is_add_like(b) else d
        tokens.append(("마플", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(b, b_d))
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
        tokens.extend(expr_to_tokens_with_pitch(a, a_d))
        tokens.append(("나누기", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(b, b_d))
        return tokens

    if isinstance(expr, Slash):  # a/b
        a, b = expr.left, expr.right
        a_d = d + 1 if is_operator(a) else d
        b_d = d + 1 if is_operator(b) else d
        # 읽기: b 분의 a
        tokens.extend(expr_to_tokens_with_pitch(b, b_d))
        tokens.append(("분의", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(a, a_d))
        return tokens

    if isinstance(expr, Frac):  # Frac(num, denom) -> denom/num
        num_part = expr.num
        denom_part = expr.denom
        num_d = d + 1 if is_operator(num_part) else d
        denom_d = d + 1 if is_operator(denom_part) else d
        # 읽기: denom 분의 num
        tokens.extend(expr_to_tokens_with_pitch(denom_part, denom_d))
        tokens.append(("분의", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(num_part, num_d))
        return tokens

    if isinstance(expr, MixedFrac):  # whole num/denom
        whole = expr.whole
        num_part = expr.num
        denom_part = expr.denom

        tokens.extend(expr_to_tokens_with_pitch(whole, d))
        tokens.append(("와", d, 0))

        denom_d = d + 1 if is_operator(denom_part) else d
        num_d = d + 1 if is_operator(num_part) else d

        tokens.extend(expr_to_tokens_with_pitch(denom_part, denom_d))
        tokens.append(("분의", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(num_part, num_d))
        return tokens

    if isinstance(expr, Mult):
        a, b = expr.left, expr.right
        a_d = d + 1 if is_add_like(a) else d
        b_d = d + 1 if is_add_mult_chain(b) else d
        tokens.extend(expr_to_tokens_with_pitch(a, a_d))
        tokens.append(("곱하기", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(b, b_d))
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

        a_tokens = expr_to_tokens_with_pitch(a, a_d)
        b_tokens = expr_to_tokens_with_pitch(b, b_d)

        # 규칙 1: 둘 다 d+1이면 b의 첫 토큰 volume +1
        if a_tokens and b_tokens:
            if a_tokens[0][1] == d + 1 and b_tokens[0][1] == d + 1:
                t, p, v = b_tokens[0]
                b_tokens[0] = (t, p, v + 1)

        # 규칙 2: 마지막과 첫 pitch가 d+2 이상이면 "곱" 삽입
        if a_tokens and b_tokens:
            last_p = a_tokens[-1][1]
            first_p = b_tokens[0][1]
            if last_p >= d + 2:
                tokens.extend(a_tokens)
                tokens.append(("곱", d, 0))
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
                tokens.extend(expr_to_tokens_with_pitch(expo, d))
                tokens.append(("제곱", d, 0))

            # 인자 x
            arg = base.arg
            arg_d = d + 1 if is_mid_or_postfix(arg) else d
            tokens.extend(expr_to_tokens_with_pitch(arg, arg_d))
            return tokens

        # --- [기본 처리] 일반적인 거듭제곱 ---
        # base pitch
        if isinstance(base, Power) or is_infix(base):
            base_d = d + 1
        else:
            base_d = d

        tokens.extend(expr_to_tokens_with_pitch(base, base_d))

        # 지수가 2면 그냥 "제곱"만
        if is_integer_value(expo) and str(expo.val) == "2":
            tokens.append(("제곱", d, 0))
        else:
            tokens.append(("의", d, 0))
            tokens.extend(expr_to_tokens_with_pitch(expo, d))
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
            tokens.extend(expr_to_tokens_with_pitch(radicand, radicand_d))
        else:
            tokens.extend(expr_to_tokens_with_pitch(index, index_d))
            tokens.append(("제곱근", d, 0))
            tokens.extend(expr_to_tokens_with_pitch(radicand, radicand_d))
        return tokens

# ==================== 비교/서술 연산자 ====================

    if isinstance(expr, Eq):
        a, b = expr.left, expr.right
        a_tokens = expr_to_tokens_with_pitch(a, d)
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
            leftmost_tokens = expr_to_tokens_with_pitch(leftmost, d)
            tokens.extend(leftmost_tokens)
            tokens.append(("이고", d, 0))
            tokens.extend(expr_to_tokens_with_pitch(b, d))
            return tokens

        # 일반 a = b : "a는 b"
        tokens.extend(expr_to_tokens_with_pitch(b, d))
        return tokens


    if isinstance(expr, Neq):
        a, b = expr.left, expr.right
        a_tokens = expr_to_tokens_with_pitch(a, d)
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
            b_head_tokens = expr_to_tokens_with_pitch(b_head_expr, d)
            tokens.extend(b_head_tokens)

            if b_head_tokens:
                head_b = pick_head_for_particle_from_tokens(b_head_tokens)
                nen2, iga2, reul2, gwa2, euro2 = get_particle(head_b)
                tokens.append((iga2, d, 0))  # "이/가"
            else:
                tokens.append(("이", d, 0))

            tokens.append(("아니고", d, 0))

            # 나머지 서술 전체 (예: 3 < 4 → "삼은 사보다 작다")
            tokens.extend(expr_to_tokens_with_pitch(b, d))
            return tokens

        # 일반 a ≠ b : "a는 b이 아니다"
        b_tokens = expr_to_tokens_with_pitch(b, d)
        tokens.extend(b_tokens)
        tokens.append(("이", d, 0))
        tokens.append(("아니다", d, 0))
        return tokens


    if isinstance(expr, Less):
        a, b = expr.left, expr.right
        a_tokens = expr_to_tokens_with_pitch(a, d)
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
            leftmost_tokens = expr_to_tokens_with_pitch(leftmost, d)
            tokens.extend(leftmost_tokens)
            tokens.append(("보다", d, 0))
            tokens.append(("작고", d, 0))
            tokens.extend(expr_to_tokens_with_pitch(b, d))
            return tokens

        # 일반 a < b : "a는 b보다 작다"
        b_tokens = expr_to_tokens_with_pitch(b, d)
        tokens.extend(b_tokens)
        tokens.append(("보다", d, 0))
        tokens.append(("작다", d, 0))
        return tokens


    if isinstance(expr, Leq):
        a, b = expr.left, expr.right
        a_tokens = expr_to_tokens_with_pitch(a, d)
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
            leftmost_tokens = expr_to_tokens_with_pitch(leftmost, d)
            tokens.extend(leftmost_tokens)
            tokens.append(("이하이고", d, 0))
            tokens.extend(expr_to_tokens_with_pitch(b, d))
            return tokens

        # 일반 a ≤ b
        tokens.extend(expr_to_tokens_with_pitch(b, d))
        tokens.append(("이하이다", d, 0))
        return tokens


    if isinstance(expr, Greater):
        a, b = expr.left, expr.right
        a_tokens = expr_to_tokens_with_pitch(a, d)
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
            leftmost_tokens = expr_to_tokens_with_pitch(leftmost, d)
            tokens.extend(leftmost_tokens)
            tokens.append(("보다", d, 0))
            tokens.append(("크고", d, 0))
            tokens.extend(expr_to_tokens_with_pitch(b, d))
            return tokens

        # 일반 a > b
        tokens.extend(expr_to_tokens_with_pitch(b, d))
        tokens.append(("보다", d, 0))
        tokens.append(("크다", d, 0))
        return tokens


    if isinstance(expr, Geq):
        a, b = expr.left, expr.right
        a_tokens = expr_to_tokens_with_pitch(a, d)
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
            leftmost_tokens = expr_to_tokens_with_pitch(leftmost, d)
            tokens.extend(leftmost_tokens)
            tokens.append(("이상이고", d, 0))
            tokens.extend(expr_to_tokens_with_pitch(b, d))
            return tokens

        # 일반 a ≥ b
        tokens.extend(expr_to_tokens_with_pitch(b, d))
        tokens.append(("이상이다", d, 0))
        return tokens

    # ==================== 집합 관련 ====================

    if isinstance(expr, SetIn):
        elem = expr.elem
        set_expr = expr.set_expr

        # 왼쪽 원소 A
        elem_tokens = expr_to_tokens_with_pitch(elem, d)
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
            leftmost_tokens = expr_to_tokens_with_pitch(leftmost, d)
            tokens.extend(leftmost_tokens)
            tokens.append(("의", d, 0))
            tokens.append(("원소이고", d, 0))
            tokens.extend(expr_to_tokens_with_pitch(set_expr, d))
            return tokens

        # 일반: "A는 B의 원소이다"
        set_tokens = expr_to_tokens_with_pitch(set_expr, d)
        tokens.extend(set_tokens)
        tokens.append(("의", d, 0))
        tokens.append(("원소이다", d, 0))
        return tokens


    if isinstance(expr, SetNotIn):
        elem = expr.elem
        set_expr = expr.set_expr

        # 왼쪽 원소 A
        elem_tokens = expr_to_tokens_with_pitch(elem, d)
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
            leftmost_tokens = expr_to_tokens_with_pitch(leftmost, d)
            tokens.extend(leftmost_tokens)
            tokens.append(("의", d, 0))
            tokens.append(("원소가", d, 0))
            tokens.append(("아니고", d, 0))
            tokens.extend(expr_to_tokens_with_pitch(set_expr, d))
            return tokens

        # 일반: "A는 B의 원소가 아니다"
        set_tokens = expr_to_tokens_with_pitch(set_expr, d)
        tokens.extend(set_tokens)
        tokens.append(("의", d, 0))
        tokens.append(("원소가", d, 0))
        tokens.append(("아니다", d, 0))
        return tokens


    if isinstance(expr, SetBuilder):
        var = expr.var
        cond = expr.condition
        tokens.append(("집합", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(var, d))
        tokens.append(("바", d, 0))
        # 조건은 살짝 강조
        tokens.extend(expr_to_tokens_with_pitch(cond, d + 1))
        return tokens


    if isinstance(expr, SetRoster):
        elements = expr.elements
        tokens.append(("집합", d, 0))
        for i, elem in enumerate(elements):
            elem_d = d + 1 if is_mid_or_postfix(elem) else d
            tokens.extend(expr_to_tokens_with_pitch(elem, elem_d))
            if i < len(elements) - 1:
                tokens.append(("콤마", d, 0))
        return tokens


    if isinstance(expr, SetNum):
        A = expr.expr
        A_d = d + 1 if is_mid_or_postfix(A) else d
        tokens.extend(expr_to_tokens_with_pitch(A, A_d))
        tokens.append(("의", d, 0))
        tokens.append(("원소의", d, 0))
        tokens.append(("개수", d, 0))
        return tokens


    if isinstance(expr, (SetSub, SetSup, SetNotSub, SetNotSup)):
        A, B = expr.left, expr.right

        # 왼쪽 집합 A
        A_tokens = expr_to_tokens_with_pitch(A, d)
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
                leftmost_tokens = expr_to_tokens_with_pitch(leftmost, d)
                tokens.extend(leftmost_tokens)
                tokens.append(("의", d, 0))
                tokens.append(("부분집합이고", d, 0))
                tokens.extend(expr_to_tokens_with_pitch(B, d))
                return tokens

            # 일반: "A는 B의 부분집합이다"
            tokens.extend(expr_to_tokens_with_pitch(B, d))
            tokens.append(("의", d, 0))
            tokens.append(("부분집합이다", d, 0))
            return tokens

        # ⊄
        if isinstance(expr, SetNotSub):
            tokens.extend(expr_to_tokens_with_pitch(B, d))
            tokens.append(("의", d, 0))
            tokens.append(("부분집합이", d, 0))
            tokens.append(("아니다", d, 0))
            return tokens

        # ⊃
        if isinstance(expr, SetSup):
            # A ⊃ (서술형) → "A는 B를 포함하고 B는 C와 같다"
            if is_descriptive_operator(B):
                leftmost = get_leftmost_operand(B)
                leftmost_tokens = expr_to_tokens_with_pitch(leftmost, d)
                tokens.extend(leftmost_tokens)
                tokens.append(("을", d, 0))
                tokens.append(("포함하고", d, 0))
                tokens.extend(expr_to_tokens_with_pitch(B, d))
                return tokens

            # 일반: "A는 B를 포함한다"
            tokens.extend(expr_to_tokens_with_pitch(B, d))
            tokens.append(("을", d, 0))
            tokens.append(("포함한다", d, 0))
            return tokens

        # ⊅
        # SetNotSup
        tokens.extend(expr_to_tokens_with_pitch(B, d))
        tokens.append(("을", d, 0))
        tokens.append(("포함하지", d, 0))
        tokens.append(("않는다", d, 0))
        return tokens

    if isinstance(expr, SetCup):
        A, B = expr.left, expr.right
        A_d = d + 1 if is_mid_or_postfix(A) or isinstance(A, SetCup) else d
        B_d = d + 1 if is_mid_or_postfix(B) else d
        tokens.extend(expr_to_tokens_with_pitch(A, A_d))
        tokens.append(("합집합", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(B, B_d))
        return tokens

    if isinstance(expr, SetCap):
        A, B = expr.left, expr.right
        A_d = d + 1 if is_mid_or_postfix(A) or isinstance(A, SetCap) else d
        B_d = d + 1 if is_mid_or_postfix(B) else d
        tokens.extend(expr_to_tokens_with_pitch(A, A_d))
        tokens.append(("교집합", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(B, B_d))
        return tokens

    # ==================== 첨자 ====================

    if isinstance(expr, Subscript):
        base = expr.base
        sub = expr.sub

        base_d = d + 1 if is_operator(base) else d
        base_tokens = expr_to_tokens_with_pitch(base, base_d)

        # sub이 정수인지 확인
        is_int = is_integer_value(sub)
        sub_d = d if is_int else d + 1
        sub_tokens = expr_to_tokens_with_pitch(sub, sub_d)

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
        base_tokens = expr_to_tokens_with_pitch(base, base_d)

        is_int = is_integer_value(sup)
        sup_d = d if is_int else d + 1
        sup_tokens = expr_to_tokens_with_pitch(sup, sup_d)

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

    if isinstance(expr, (Rimpl, Limpl, Rsufficient, Lsufficient)):
        p, q = expr.left, expr.right
        p_d = d + 1 if is_mid_or_postfix(p) else d
        q_d = d + 1 if is_mid_or_postfix(q) else d

        # 왼쪽 명제 p 먼저 읽기
        p_tokens = expr_to_tokens_with_pitch(p, p_d)
        tokens.extend(p_tokens)

        # 왼쪽 명제에 붙일 조사 계산용
        if p_tokens:
            head = pick_head_for_particle_from_tokens(p_tokens)
            nen, iga, reul, gwa, euro = get_particle(head)
        else:
            # p가 비어 있는 극단적인 경우 기본값
            nen, iga, reul, gwa, euro = ("는", "이", "을", "과", "으로")

        # 오른쪽 명제 토큰
        q_tokens = expr_to_tokens_with_pitch(q, q_d)

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
        tokens.extend(expr_to_tokens_with_pitch(symbol, d))
        # tokens.append(("콜론", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(statement, d))
        return tokens

    if isinstance(expr, FuncDef):
        func = expr.func
        mapping = expr.mapping

        # "함수" + 함수 이름
        tokens.append(("함수", d, 0))
        func_tokens = expr_to_tokens_with_pitch(func, d)
        tokens.extend(func_tokens)

        # 함수 이름에 맞는 "은/는"
        if func_tokens:
            func_head = func_tokens[-1][0]
            nen, iga, reul, gwa, euro = get_particle(func_head)
        else:
            nen, iga, reul, gwa, euro = ("는", "이", "을", "과", "으로")
        tokens.append((nen, d, 0))   # 는/은

        # 치역(또는 대응집합) 쪽 표현
        mapping_tokens = expr_to_tokens_with_pitch(mapping, d)
        tokens.extend(mapping_tokens)

        # mapping 뒤에 붙일 "으로/로"
        if mapping_tokens:
            mapping_head = mapping_tokens[-1][0]
            _, _, _, _, euro2 = get_particle(mapping_head)
        else:
            euro2 = "으로"
        tokens.append((euro2, d, 0))  # 으로/로

        tokens.append(("가는", d, 0))
        tokens.append(("함수", d, 0))
        return tokens


    # ==================== 함수 ====================

    if isinstance(expr, Func):
        name = expr.name
        args = expr.args

        name_d = d + 1 if is_mid_or_postfix(name) else d
        tokens.extend(expr_to_tokens_with_pitch(name, name_d))

        for i, arg in enumerate(args):
            arg_d = d + 1 if is_mid_or_postfix(arg) else d
            tokens.extend(expr_to_tokens_with_pitch(arg, arg_d))
            if i < len(args) - 1:
                tokens.append(("콤마", d, 0))

        return tokens

    if isinstance(expr, FuncInv):
        name = expr.name
        args = expr.args

        name_d = d + 1 if is_operator(name) else d
        tokens.extend(expr_to_tokens_with_pitch(name, name_d))
        tokens.append(("인버스", d, 0))

        for i, arg in enumerate(args):
            arg_d = d + 1 if is_mid_or_postfix(arg) else d
            tokens.extend(expr_to_tokens_with_pitch(arg, arg_d))
            if i < len(args) - 1:
                tokens.append(("콤마", d, 0))

        return tokens

    # ==================== 조합론 ====================

    if isinstance(expr, Perm):
        n, r = expr.n, expr.r
        n_d = d + 1 if is_mid_or_postfix(n) else d
        r_d = d + 1 if is_mid_or_postfix(r) else d
        tokens.append(("순열", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(n, n_d))
        tokens.append(("피", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(r, r_d))
        return tokens

    if isinstance(expr, Comb):
        n, r = expr.n, expr.r
        n_d = d + 1 if is_mid_or_postfix(n) else d
        r_d = d + 1 if is_mid_or_postfix(r) else d
        tokens.append(("조합", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(n, n_d))
        tokens.append(("씨", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(r, r_d))
        return tokens

    if isinstance(expr, RepeatedPermu):
        n, r = expr.n, expr.r
        n_d = d + 1 if is_mid_or_postfix(n) else d
        r_d = d + 1 if is_mid_or_postfix(r) else d
        tokens.append(("중복순열", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(n, n_d))
        tokens.append(("파이", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(r, r_d))
        return tokens

    if isinstance(expr, RepeatedComb):
        n, r = expr.n, expr.r
        n_d = d + 1 if is_mid_or_postfix(n) else d
        r_d = d + 1 if is_mid_or_postfix(r) else d
        tokens.append(("중복조합", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(n, n_d))
        tokens.append(("에이치", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(r, r_d))
        return tokens

    # ==================== 로그/삼각함수 ====================

    if isinstance(expr, Log):
        base = expr.base
        arg = expr.arg

        arg_d = d + 1 if is_mid_or_postfix(arg) else d

        tokens.append(("로그", d, 0))

        if not isinstance(base, None_):
            tokens.extend(expr_to_tokens_with_pitch(base, d))
            tokens.append(("의", d, 0))

        tokens.extend(expr_to_tokens_with_pitch(arg, arg_d))
        return tokens

    if isinstance(expr, Ln):
        arg = expr.arg
        arg_d = d + 1 if is_mid_or_postfix(arg) else d
        tokens.append(("엘엔", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(arg, arg_d))
        return tokens

    if isinstance(expr, Sin):
        arg = expr.arg
        arg_d = d + 1 if is_mid_or_postfix(arg) else d
        tokens.append(("사인", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(arg, arg_d))
        return tokens

    if isinstance(expr, Cos):
        arg = expr.arg
        arg_d = d + 1 if is_mid_or_postfix(arg) else d
        tokens.append(("코사인", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(arg, arg_d))
        return tokens

    if isinstance(expr, Tan):
        arg = expr.arg
        arg_d = d + 1 if is_mid_or_postfix(arg) else d
        tokens.append(("탄젠트", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(arg, arg_d))
        return tokens

    if isinstance(expr, Sec):
        arg = expr.arg
        arg_d = d + 1 if is_mid_or_postfix(arg) else d
        tokens.append(("시컨트", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(arg, arg_d))
        return tokens

    if isinstance(expr, Csc):
        arg = expr.arg
        arg_d = d + 1 if is_mid_or_postfix(arg) else d
        tokens.append(("코시컨트", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(arg, arg_d))
        return tokens

    if isinstance(expr, Cot):
        arg = expr.arg
        arg_d = d + 1 if is_mid_or_postfix(arg) else d
        tokens.append(("코탄젠트", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(arg, arg_d))
        return tokens

    # ==================== 수열/극한/급수/미적분 ====================

    if isinstance(expr, Seq):
        term = expr.term
        tokens.append(("수열", d, 0))
        # term이 Subscript이든, Value이든, 다른 거든 전부 공평하게 재귀 처리
        tokens.extend(expr_to_tokens_with_pitch(term, d))
        return tokens

    if isinstance(expr, Lim):
        var = expr.var
        to = expr.to
        expr_inner = expr.expr

        tokens.append(("리미트", d, 0))

        # 변수 부분: n이, x가 ...
        var_tokens = expr_to_tokens_with_pitch(var, d)
        tokens.extend(var_tokens)
        if var_tokens:
            var_head = var_tokens[-1][0]
            _, iga, _, _, _ = get_particle(var_head)
        else:
            iga = "이"
        tokens.append((iga, d, 0))   # 이/가

        # → 값 부분: 무한대로, 0으로 ...
        to_tokens = expr_to_tokens_with_pitch(to, d)
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
        tokens.extend(expr_to_tokens_with_pitch(expr_inner, expr_d))
        return tokens

    if isinstance(expr, Sum):
        term = expr.term
        var = expr.var
        start = expr.start
        end = expr.end

        tokens.append(("시그마", d, 0))

        # 시그마 아래 변수: n은, k는 ...
        var_tokens = expr_to_tokens_with_pitch(var, d)
        tokens.extend(var_tokens)
        if var_tokens:
            var_head = var_tokens[-1][0]
            nen, _, _, _, _ = get_particle(var_head)
        else:
            nen = "는"
        tokens.append((nen, d, 0))   # 은/는

        # 범위: a부터 b까지
        tokens.extend(expr_to_tokens_with_pitch(start, d))
        tokens.append(("부터", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(end, d))
        tokens.append(("까지", d, 0))

        # 합의 항
        term_d = d + 1 if is_mid_or_postfix(term) else d
        tokens.extend(expr_to_tokens_with_pitch(term, term_d))
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
            tokens.extend(expr_to_tokens_with_pitch(n, d))
            tokens.append(("제곱", d, 0))

        # y 부분
        y_d = d + 1 if is_mid_or_postfix(y) else d
        tokens.extend(expr_to_tokens_with_pitch(y, y_d))

        # 중간 "디"
        tokens.append(("디", d, 0))

        # x 부분
        x_d = d + 1 if is_mid_or_postfix(x) else d
        tokens.extend(expr_to_tokens_with_pitch(x, x_d))

        # 뒤쪽 지수
        if n_str == "1":
            pass
        elif n_str == "2":
            tokens.append(("제곱", d, 0))
        else:
            tokens.extend(expr_to_tokens_with_pitch(n, d))
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
            tokens.extend(expr_to_tokens_with_pitch(lower, d))
            tokens.append(("부터", d, 0))

        if not isinstance(upper, None_):
            tokens.extend(expr_to_tokens_with_pitch(upper, d))
            tokens.append(("까지", d, 0))

        integrand_d = d + 1 if is_mid_or_postfix(integrand) else d
        tokens.extend(expr_to_tokens_with_pitch(integrand, integrand_d))

        tokens.append(("디", d, 0))
        var_d = d + 1 if is_mid_or_postfix(var) else d
        tokens.extend(expr_to_tokens_with_pitch(var, var_d))

        return tokens

    if isinstance(expr, Integrated):
        antiderivative = expr.antiderivative
        lower = expr.lower
        upper = expr.upper

        tokens.extend(expr_to_tokens_with_pitch(antiderivative, d + 1))
        tokens.extend(expr_to_tokens_with_pitch(lower, d))
        tokens.append(("부터", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(upper, d))
        tokens.append(("까지", d, 0))
        return tokens

    # ================== 구간 ==================
    if isinstance(expr, ClosedInterval):
        tokens.extend(expr_to_tokens_with_pitch(expr.left, d))
        tokens.append(("초과", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(expr.right, d))
        tokens.append(("미만", d, 0))
        return tokens

    if isinstance(expr, ClosedOpenInterval):
        tokens.extend(expr_to_tokens_with_pitch(expr.left, d))
        tokens.append(("초과", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(expr.right, d))
        tokens.append(("이하", d, 0))
        return tokens

    if isinstance(expr, OpenClosedInterval):
        tokens.extend(expr_to_tokens_with_pitch(expr.left, d))
        tokens.append(("이상", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(expr.right, d))
        tokens.append(("미만", d, 0))
        return tokens

    if isinstance(expr, OpenInterval):
        tokens.extend(expr_to_tokens_with_pitch(expr.left, d))
        tokens.append(("이상", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(expr.right, d))
        tokens.append(("이하", d, 0))
        return tokens

    # ==================== 확률 ====================

    if isinstance(expr, Prob):
        event = expr.event
        condition = expr.condition

        tokens.append(("확률", d, 0))
        tokens.append(("피", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(event, d))

        if condition is not None and not isinstance(condition, None_):
            tokens.append(("바", d, 0))
            tokens.extend(expr_to_tokens_with_pitch(condition, d))

        return tokens

    # ==================== 기하 ====================

    if isinstance(expr, Triangle):
        vertices = expr.vertices
        tokens.append(("삼각형", d, 0))
        # vertices는 문자열이므로 Text로 변환
        tokens.extend(expr_to_tokens_with_pitch(Text(vertices), d))
        return tokens

    if isinstance(expr, Angle):
        vertices = expr.vertices
        tokens.append(("각", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(Text(vertices), d))
        return tokens

    if isinstance(expr, Arc):
        vertices = expr.vertices
        tokens.append(("호", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(Text(vertices), d))
        return tokens

    if isinstance(expr, Point):
        name = expr.name
        args = expr.args

        if name is not None and not isinstance(name, None_):
            tokens.append(("점", d, 0))
            tokens.extend(expr_to_tokens_with_pitch(name, d))

        for i, arg in enumerate(args):
            arg_d = d + 1 if is_operator(arg) else d
            tokens.extend(expr_to_tokens_with_pitch(arg, arg_d))
            if i < len(args) - 1:
                tokens.append(("콤마", d, 0))

        return tokens

    if isinstance(expr, Segment):
        start = expr.start
        end = expr.end
        tokens.append(("선분", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(start, d))
        tokens.extend(expr_to_tokens_with_pitch(end, d))
        return tokens

    if isinstance(expr, Ray):
        start = expr.start
        through = expr.through
        tokens.append(("반직선", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(start, d))
        tokens.extend(expr_to_tokens_with_pitch(through, d))
        return tokens

    if isinstance(expr, Line):
        point1 = expr.point1
        point2 = expr.point2
        tokens.append(("직선", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(point1, d))
        tokens.extend(expr_to_tokens_with_pitch(point2, d))
        return tokens

    if isinstance(expr, LineExpr):
        line = expr.line
        eq = expr.eq
        tokens.append(("직선", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(line, d))
        # tokens.append(("콜론", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(eq, d))
        return tokens

    if isinstance(expr, Perp):
        line1 = expr.line1
        line2 = expr.line2

        # 첫 번째 직선
        line1_tokens = expr_to_tokens_with_pitch(line1, d)
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
            left2_tokens = expr_to_tokens_with_pitch(left2, d)
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
            tokens.extend(expr_to_tokens_with_pitch(line2, d))
        else:
            # 단일 관계: "수직이다"
            line2_tokens = expr_to_tokens_with_pitch(line2, d)
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

        line1_tokens = expr_to_tokens_with_pitch(line1, d)
        tokens.extend(line1_tokens)

        if line1_tokens:
            last1 = line1_tokens[-1][0]
            _, _, _, wa_gwa, _ = get_particle(last1)
        else:
            wa_gwa = "와"
        tokens.append((wa_gwa, d, 0))

        if is_descriptive_operator(line2):
            left2 = get_leftmost_operand(line2)
            left2_tokens = expr_to_tokens_with_pitch(left2, d)
            tokens.extend(left2_tokens)

            if left2_tokens:
                last2 = left2_tokens[-1][0]
                eun_neun, _, _, _, _ = get_particle(last2)
            else:
                eun_neun = "은"
            tokens.append((eun_neun, d, 0))
            tokens.append(("평행이고", d, 0))
            tokens.extend(expr_to_tokens_with_pitch(line2, d))
        else:
            line2_tokens = expr_to_tokens_with_pitch(line2, d)
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
        left = expr.left
        right = expr.right
        tokens.extend(expr_to_tokens_with_pitch(left, d))
        tokens.append(("대", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(right, d))
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
        tokens.extend(expr_to_tokens_with_pitch(vec1, vec1_d))
        tokens.append(("내적", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(vec2, vec2_d))
        return tokens

    if isinstance(expr, Congru):
        shape1 = expr.shape1
        shape2 = expr.shape2

        shape1_tokens = expr_to_tokens_with_pitch(shape1, d)
        tokens.extend(shape1_tokens)

        if shape1_tokens:
            last1 = shape1_tokens[-1][0]
            _, _, _, wa_gwa, _ = get_particle(last1)
        else:
            wa_gwa = "와"
        tokens.append((wa_gwa, d, 0))

        if is_descriptive_operator(shape2):
            left2 = get_leftmost_operand(shape2)
            left2_tokens = expr_to_tokens_with_pitch(left2, d)
            tokens.extend(left2_tokens)

            if left2_tokens:
                last2 = left2_tokens[-1][0]
                eun_neun, _, _, _, _ = get_particle(last2)
            else:
                eun_neun = "은"
            tokens.append((eun_neun, d, 0))
            tokens.append(("합동이고", d, 0))
            tokens.extend(expr_to_tokens_with_pitch(shape2, d))
        else:
            shape2_tokens = expr_to_tokens_with_pitch(shape2, d)
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

        shape1_tokens = expr_to_tokens_with_pitch(shape1, d)
        tokens.extend(shape1_tokens)

        if shape1_tokens:
            last1 = shape1_tokens[-1][0]
            _, _, _, wa_gwa, _ = get_particle(last1)
        else:
            wa_gwa = "와"
        tokens.append((wa_gwa, d, 0))

        if is_descriptive_operator(shape2):
            left2 = get_leftmost_operand(shape2)
            left2_tokens = expr_to_tokens_with_pitch(left2, d)
            tokens.extend(left2_tokens)

            if left2_tokens:
                last2 = left2_tokens[-1][0]
                eun_neun, _, _, _, _ = get_particle(last2)
            else:
                eun_neun = "은"
            tokens.append((eun_neun, d, 0))
            tokens.append(("닮음이고", d, 0))
            tokens.extend(expr_to_tokens_with_pitch(shape2, d))
        else:
            shape2_tokens = expr_to_tokens_with_pitch(shape2, d)
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
        kor = unit_to_korean(expr.unit)
        return [(kor, d, 0)]

    if isinstance(expr, UnitDiv):
        num_unit = expr.num_unit
        denom_unit = expr.denom_unit
        tokens.extend(expr_to_tokens_with_pitch(num_unit, d))
        tokens.append(("퍼", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(denom_unit, d))
        return tokens

    if isinstance(expr, UnitMult):
        left = expr.left
        right = expr.right
        tokens.extend(expr_to_tokens_with_pitch(left, d))
        tokens.extend(expr_to_tokens_with_pitch(right, d))
        return tokens

    if isinstance(expr, About):
        inner = expr.expr
        inner_d = d + 1 if is_mid_or_postfix(inner) else d
        tokens.append(("약", d, 0))
        tokens.extend(expr_to_tokens_with_pitch(inner, inner_d))
        return tokens

    # ==================== Cases (연립방정식) ====================

    if isinstance(expr, Cases):
        for i, (case_expr, cond) in enumerate(expr.cases):
            # 1) condition을 현재 깊이 d에서 토큰화
            cond_tokens = expr_to_tokens_with_pitch(cond, d)

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
            tokens.extend(expr_to_tokens_with_pitch(case_expr, d))

            # 3) 마지막 case가 아니면 "그리고" 추가
            if i < len(expr.cases) - 1:
                tokens.append(("그리고", d, 0))

        return tokens

    # ==================== InlinePhrase ====================

    if isinstance(expr, InlinePhrase):
        for part in expr.parts:
            tokens.extend(expr_to_tokens_with_pitch(part, d))
        return tokens

    # ==================== Fallback ====================

    # 정의되지 않은 경우
    return [(str(expr), d, 0)]


# ===================== 테스트 함수 =====================

def test_pitch_calculation():
    """테스트 케이스로 pitch level 검증"""

    print("=" * 60)
    print("Pitch Level 테스트")
    print("=" * 60)

    # 테스트 1: 2와 4/5 ÷ 4
    print("\n테스트 1: 2와 4/5 ÷ 4")
    expr1 = Divide(
        MixedFrac(Value(2), Value(4), Value(5)),
        Value(4)
    )
    tokens1 = expr_to_tokens_with_pitch(expr1)
    print("토큰:", tokens1)

    # 테스트 2: 2와 (4÷4)/5
    print("\n테스트 2: 2와 (4÷4)/5")
    expr2 = MixedFrac(
        Value(2),
        Divide(Value(4), Value(4)),
        Value(5)
    )
    tokens2 = expr_to_tokens_with_pitch(expr2)
    print("토큰:", tokens2)

    # 테스트 3: 10 < 10÷1/□ < 20
    print("\n테스트 3: 10 < 10÷1/□ < 20")
    expr3 = Less(
        Value(10),
        Less(
            Divide(Value(10), Slash(Value(1), Square())),
            Value(20)
        )
    )
    tokens3 = expr_to_tokens_with_pitch(expr3)
    print("토큰:", tokens3)

    # 테스트 4: lim_(n→∞) (1+1/n)^n
    print("\n테스트 4: lim_(n→∞) (1+1/n)^n")
    expr4 = Lim(
        Value('n'),
        Infty(),
        Power(
            Add(Value(1), Frac(Value(1), Value('n'))),
            Value('n')
        )
    )
    tokens4 = expr_to_tokens_with_pitch(expr4)
    print("토큰:", tokens4)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_pitch_calculation()