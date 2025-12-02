from LaTeX_Parser import latex_to_expression
from Expression_Syntax import expression_to_korean, expression_to_tokens_with_pitch
from speech_synthesizer import MathSpeechSynthesizer
from gtts_expr_audio_pitch import AudioPolicy, GTTSCache, trim_silence, silence, _ensure_pcm, pitch_shift_to_hz

import os
import sys
from typing import List, Tuple

# IPython.display는 선택적 임포트 (Jupyter/Colab 환경에서만 필요)
try:
    from IPython.display import Audio, display, HTML
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    # Streamlit 환경에서는 사용하지 않으므로 더미 객체 생성
    class Audio:
        def __init__(self, *args, **kwargs):
            pass
    def display(*args, **kwargs):
        pass
    class HTML:
        def __init__(self, *args, **kwargs):
            pass

# pydub의 audioop 의존성 문제 해결 (Python 3.13+)
try:
    import audioop
except ImportError:
    try:
        import pyaudioop as audioop
    except ImportError:
        # audioop가 없으면 더미 모듈 생성
        class DummyAudioop:
            def __getattr__(self, name):
                def dummy_func(*args, **kwargs):
                    raise NotImplementedError(f"audioop.{name} is not available")
                return dummy_func
        audioop = DummyAudioop()
        sys.modules['audioop'] = audioop
        sys.modules['pyaudioop'] = audioop

from pydub import AudioSegment

class PitchLevelMapper:
    """
    Pitch Level을 실제 Hz 값으로 변환하는 클래스
    AudioPolicy의 음높이 설정을 따름
    """
    def __init__(self, policy: AudioPolicy = None):
        self.policy = policy or AudioPolicy()

        # Pitch level to Hz 매핑
        # level 0이 기준 (C4, 261.63 Hz)
        # 양수 level은 깊어짐 (낮은 음)
        # 음수 level은 얕아짐 (높은 음)
        self.level_to_hz = {
            0: self.policy.pitch_normal,        # C4 (261.63 Hz) - 기준
            1: self.policy.pitch_increase_2,    # A3 (220.0 Hz) - 한 단계 깊게
            2: self.policy.pitch_increase_3,    # F3 (174.61 Hz) - 두 단계 깊게
            3: self.policy.pitch_increase_4,    # D3 (146.83 Hz) - 세 단계 깊게
            4: self.policy.pitch_increase_5,    # B2 (123.47 Hz) - 네 단계 깊게
            -1: self.policy.pitch_decrease_2,   # E4 (329.63 Hz) - 한 단계 얕게
            -2: self.policy.pitch_decrease_3,   # G4 (392.0 Hz) - 두 단계 얕게
            -3: self.policy.pitch_decrease_4,   # B4 (493.88 Hz) - 세 단계 얕게
            -4: self.policy.pitch_decrease_5,   # D5 (587.33 Hz) - 네 단계 얕게
        }

    def get_hz(self, level: int) -> float:
        """Pitch level을 Hz로 변환"""
        # level이 범위를 벗어나면 가장 극단값 사용
        if level > 4:
            level = 4
        elif level < -4:
            level = -4
        return self.level_to_hz[level]

    def get_note_name(self, level: int) -> str:
        """Pitch level에 해당하는 음계 이름 반환"""
        note_map = {
            0: "C4", 1: "A3", 2: "F3", 3: "D3", 4: "B2",
            -1: "E4", -2: "G4", -3: "B4", -4: "D5"
        }
        if level > 4:
            level = 4
        elif level < -4:
            level = -4
        return note_map[level]

# 인스턴스 생성
mapper = PitchLevelMapper()


class PitchVolumeLevelSynthesizer:
    """
    Pitch / Volume Level을 기반으로 음성을 합성하는 클래스.
    토큰은 (token, pitch_level[, volume_level]) 형태를 지원하며,
    volume_level이 생략되면 0으로 처리한다.
    """

    def __init__(self, policy: AudioPolicy = None, tmpdir: str = None):
        self.policy = policy or AudioPolicy()
        self.cache = GTTSCache(tmpdir=tmpdir)
        self.mapper = PitchLevelMapper(policy=self.policy)

    def synthesize(
        self,
        tokens_with_pitch: List[Tuple[str, int]]  # (token, pitch) 또는 (token, pitch, volume)
    ) -> AudioSegment:
        """
        (토큰, pitch_level[, volume_level]) 리스트를 받아 음성 합성.

        Parameters
        ----------
        tokens_with_pitch : List[Tuple[str, int]] 또는 List[Tuple[str, int, int]]
            - 필수: 토큰 문자열, pitch level
            - 선택: volume level (생략 시 0)

        Returns
        -------
        AudioSegment
            합성된 오디오
        """
        SR = 22050
        out = AudioSegment.silent(0, frame_rate=SR)

        normalized_tokens: List[Tuple[str, int, int]] = []
        for entry in tokens_with_pitch:
            if len(entry) == 3:
                token, pitch_level, volume_level = entry
            elif len(entry) == 2:
                token, pitch_level = entry
                volume_level = 0
            else:
                raise ValueError(
                    "토큰은 (token, pitch_level) 또는 (token, pitch_level, volume_level) 형식을 따라야 합니다."
                )
            normalized_tokens.append((token, pitch_level, volume_level))

        for idx, (token, pitch_level, volume_level) in enumerate(normalized_tokens):
            if idx > 0:
                out += silence(10)  # 토큰 간 10ms 간격

            speech = self.cache.tts_to_segment(token)
            speech = trim_silence(speech, threshold_db=-40, min_silence_len=50, pad_ms=6)

            target_hz = self.mapper.get_hz(pitch_level)
            speech_pitched = pitch_shift_to_hz(speech, target_hz=target_hz, base_hz=261.63)

            gain_db = volume_level * 10
            if gain_db:
                speech_pitched = speech_pitched.apply_gain(gain_db)

            out += speech_pitched

        return out

    def save(self, segment: AudioSegment, path: str) -> str:
        """오디오를 파일로 저장"""
        ext = os.path.splitext(path)[1].lower()
        if ext not in {".mp3", ".wav"}:
            path += ".mp3"
            ext = ".mp3"

        seg = _ensure_pcm(segment)
        if ext == ".mp3":
            seg.export(path, format="mp3", bitrate="192k")
        else:
            seg.export(path, format="wav")
        return path

    def play(self, segment: AudioSegment):
        """오디오를 재생 (Jupyter 환경)"""
        seg = _ensure_pcm(segment)
        display(Audio(seg.raw_data, rate=seg.frame_rate))

# 합성기 인스턴스 생성
synthesizer = PitchVolumeLevelSynthesizer()

def latex_audio_grouping_pitch(expression, output_path="latex_to_audio/single_example"):
    tokens_pitch_volume = expression_to_tokens_with_pitch(expression)
    '''
    token_display = [f"{tok}({lvl:+d})" for tok, lvl, _ in tokens_pitch_volume]
    
    print("토큰:", " ".join(token_display))

    print("\n→ 음성 합성 중...", end=" ")
    '''
    audio = synthesizer.synthesize(tokens_pitch_volume)
    
    synthesizer.save(audio, output_path)
    
