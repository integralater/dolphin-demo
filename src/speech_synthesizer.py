# speech_synthesizer.py

# -*- coding: utf-8 -*-
"""
Expression 객체를 음성으로 변환하는 통합 모듈
- Expression 객체를 입력받아 depth 정보와 함께 한글 토큰 추출
- gtts_expr_audio_pitch 모듈을 사용하여 피치 변조된 음성 생성
- 직접 재생 및 파일 저장 기능 제공
"""

from Expression_Syntax import expression_to_korean_with_depth
from gtts_expr_audio_pitch import (
    ExpressionAudioReader, 
    AudioPolicy,
    play_tokens,
    save_tokens
)
from typing import Optional


class MathSpeechSynthesizer:
    """
    수식 음성 합성기
    
    Expression 객체를 받아 시각장애인을 위한 음성을 생성합니다.
    - depth 증가(깊어짐): 낮은 음
    - depth 감소(얕아짐): 높은 음
    - |Δdepth| ≥ 2: 공백 증가 + 피치 변조
    - |Δdepth| = 1: 변조 없이 자연스럽게 연결
    """
    
    def __init__(self, policy: Optional[AudioPolicy] = None):
        """
        Parameters
        ----------
        policy : AudioPolicy, optional
            음성 합성 정책 (간격, 피치 변조 등)
            기본값: AudioPolicy(speech_rate=1.0)
        """
        self.policy = policy or AudioPolicy()
        self.reader = ExpressionAudioReader(policy=self.policy)
    
    def expression_to_tokens(self, expr) -> list:
        """
        Expression 객체를 (토큰, depth) 리스트로 변환
        
        Parameters
        ----------
        expr : Expression
            변환할 수식 객체
            
        Returns
        -------
        list of tuple
            [(토큰, depth), ...] 형식의 리스트
            
        Examples
        --------
        >>> expr = Add(Value("x"), Frac(Value(1), Value("y")))
        >>> tokens = synthesizer.expression_to_tokens(expr)
        >>> print(tokens)
        [('엑스', 1), ('더하기', 0), ('일', 2), ('분의', 1), ('와이', 2)]
        """
        return expression_to_korean_with_depth(expr)
    
    def play(self, expr):
        """
        Expression 객체를 바로 재생
        
        Parameters
        ----------
        expr : Expression
            재생할 수식 객체
            
        Returns
        -------
        tuple
            (reader, audio_segment) - 추가 조작을 위해 반환
            
        Examples
        --------
        >>> from Expression_Syntax import *
        >>> expr = Add(Value("x"), Frac(Value(1), Value("y")))
        >>> synthesizer = MathSpeechSynthesizer()
        >>> synthesizer.play(expr)
        """
        tokens = self.expression_to_tokens(expr)
        seg = self.reader.synthesize(tokens)
        self.reader.play(seg)
        return self.reader, seg
    
    def save(self, expr, output_path: str = "math_expression.mp3") -> str:
        """
        Expression 객체를 파일로 저장
        
        Parameters
        ----------
        expr : Expression
            저장할 수식 객체
        output_path : str, optional
            저장할 파일 경로 (.mp3 또는 .wav)
            기본값: "math_expression.mp3"
            
        Returns
        -------
        str
            저장된 파일 경로
            
        Examples
        --------
        >>> expr = Add(Value("x"), Frac(Value(1), Value("y")))
        >>> synthesizer = MathSpeechSynthesizer()
        >>> path = synthesizer.save(expr, "my_expression.mp3")
        >>> print(f"저장됨: {path}")
        """
        tokens = self.expression_to_tokens(expr)
        seg = self.reader.synthesize(tokens)
        return self.reader.save(seg, output_path)
    
    def play_and_save(self, expr, output_path: str = "math_expression.mp3"):
        """
        Expression 객체를 재생하고 동시에 파일로 저장
        
        Parameters
        ----------
        expr : Expression
            재생하고 저장할 수식 객체
        output_path : str, optional
            저장할 파일 경로
            
        Returns
        -------
        str
            저장된 파일 경로
        """
        tokens = self.expression_to_tokens(expr)
        seg = self.reader.synthesize(tokens)
        self.reader.play(seg)
        return self.reader.save(seg, output_path)


# ===== 편의 함수 =====

def play_expression(expr, policy: Optional[AudioPolicy] = None):
    """
    Expression 객체를 바로 재생하는 편의 함수
    
    Parameters
    ----------
    expr : Expression
        재생할 수식 객체
    policy : AudioPolicy, optional
        음성 합성 정책
        
    Examples
    --------
    >>> from Expression_Syntax import *
    >>> expr = Add(Value("x"), Value("y"))
    >>> play_expression(expr)
    """
    synthesizer = MathSpeechSynthesizer(policy=policy)
    return synthesizer.play(expr)


def save_expression(expr, output_path: str = "math_expression.mp3", 
                   policy: Optional[AudioPolicy] = None) -> str:
    """
    Expression 객체를 파일로 저장하는 편의 함수
    
    Parameters
    ----------
    expr : Expression
        저장할 수식 객체
    output_path : str, optional
        저장할 파일 경로
    policy : AudioPolicy, optional
        음성 합성 정책
        
    Returns
    -------
    str
        저장된 파일 경로
        
    Examples
    --------
    >>> expr = Frac(Value(1), Add(Value("x"), Value("y")))
    >>> path = save_expression(expr, "fraction.mp3")
    """
    synthesizer = MathSpeechSynthesizer(policy=policy)
    return synthesizer.save(expr, output_path)


# ===== 사용 예시 =====
if __name__ == "__main__":
    from Expression_Syntax import *
    
    # 예시 1: x + 1/y vs 1/(x+y)
    print("=" * 60)
    print("예시 1: 서로 다른 수식이 같은 텍스트로 읽히는 경우")
    print("=" * 60)
    
    expr1 = Add(Value("x"), Frac(Value(1), Value("y")))
    expr2 = Frac(Value(1), Add(Value("x"), Value("y")))
    
    print("\n표현식 1: x + 1/y")
    synthesizer = MathSpeechSynthesizer()
    tokens1 = synthesizer.expression_to_tokens(expr1)
    print(f"토큰: {tokens1}")
    
    print("\n표현식 2: 1/(x+y)")
    tokens2 = synthesizer.expression_to_tokens(expr2)
    print(f"토큰: {tokens2}")
    
    print("\n→ 토큰 텍스트는 같지만 depth가 다릅니다!")
    print("  이제 음성을 들어보세요. 두 수식이 다르게 들립니다.\n")
    
    # 파일로 저장
    path1 = synthesizer.save(expr1, "example1_x_plus_1_over_y.mp3")
    print(f"✓ 저장됨: {path1}")
    
    path2 = synthesizer.save(expr2, "example2_1_over_x_plus_y.mp3")
    print(f"✓ 저장됨: {path2}")
    
    # 예시 2: 복잡한 수식
    print("\n" + "=" * 60)
    print("예시 2: 복잡한 중첩 구조")
    print("=" * 60)
    
    # sin²(x) + y + z vs sin²(x+y+z)
    expr3 = Add(Add(Pow(Sin(Value("x")), Value(2)), Value("y")), Value("z"))
    expr4 = Pow(Sin(Add(Add(Value("x"), Value("y")), Value("z"))), Value(2))
    
    print("\n표현식 3: sin²(x) + y + z")
    tokens3 = synthesizer.expression_to_tokens(expr3)
    print(f"토큰: {tokens3}")
    path3 = synthesizer.save(expr3, "example3_sin2x_plus_y_plus_z.mp3")
    print(f"✓ 저장됨: {path3}")
    
    print("\n표현식 4: sin²(x+y+z)")
    tokens4 = synthesizer.expression_to_tokens(expr4)
    print(f"토큰: {tokens4}")
    path4 = synthesizer.save(expr4, "example4_sin2_xyz.mp3")
    print(f"✓ 저장됨: {path4}")
    
    print("\n" + "=" * 60)
    print("모든 오디오 파일이 생성되었습니다!")
    print("각 파일을 들어보시고 depth 변화에 따른 피치 변조를 확인하세요.")
    print("=" * 60)

