# gtts_expr_audio_pitch.py
# -*- coding: utf-8 -*-
"""
Colab용 (token, depth) -> gTTS 합성/재생/저장, 이어콘 없이 '피치 변조'로 깊이 표현
- Δdepth >= +3: 피치 다운(낮게) - 올라갈 때는 3부터
- Δdepth <= -2: 피치 UP(높게) - 내려갈 때는 2부터
- |Δdepth| < threshold: 변조 없음
- 토큰 오디오 앞/뒤 침묵 자동 트림
- 재생: IPython.display.Audio
- 저장: mp3 또는 wav
"""

import os, time, hashlib, tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import numpy as np
from gtts import gTTS
from pydub import AudioSegment, effects
from pydub.silence import detect_nonsilent
from pydub import utils as pydub_utils
from IPython.display import Audio, display

# ffmpeg 경로 설정 (imageio-ffmpeg 사용)
import subprocess
try:
    from imageio_ffmpeg import get_ffmpeg_exe
    ffmpeg_exe = get_ffmpeg_exe()
    
    # AudioSegment 클래스 속성 설정
    AudioSegment.converter = ffmpeg_exe
    AudioSegment.ffmpeg = ffmpeg_exe
    AudioSegment.ffprobe = ffmpeg_exe
    
    # pydub.utils.which 함수를 오버라이드 (이게 핵심!)
    original_which = pydub_utils.which
    def patched_which(program):
        if program in ["ffmpeg", "ffprobe", "avconv", "avprobe"]:
            return ffmpeg_exe
        return original_which(program)
    pydub_utils.which = patched_which
    
    print(f"✓ pydub ffmpeg 경로 설정됨: {ffmpeg_exe}")
    print(f"✓ pydub.utils.which 패치됨")
except ImportError:
    print("✗ imageio-ffmpeg 없음 - 시스템 ffmpeg 사용")
except Exception as e:
    print(f"✗ ffmpeg 설정 오류: {e}")

# ===== 기본 오디오 파라미터 =====
SR = 22050
SAMPLE_WIDTH = 2
CHANNELS = 1

def _ensure_pcm(seg: AudioSegment) -> AudioSegment:
    return seg.set_frame_rate(SR).set_channels(CHANNELS).set_sample_width(SAMPLE_WIDTH)

def silence(dur_ms: int) -> AudioSegment:
    return AudioSegment.silent(duration=max(0, dur_ms), frame_rate=SR)

# ---- 피치 변조 (Hz 기반) ----
def pitch_shift_to_hz(seg: AudioSegment, target_hz: float, base_hz: float = 261.63) -> AudioSegment:
    """
    특정 Hz로 피치를 변조 (길이 유지)
    
    Parameters
    ----------
    seg : AudioSegment
        변조할 오디오 세그먼트
    target_hz : float
        목표 주파수 (Hz)
    base_hz : float
        원본 주파수 (Hz), 기본값은 C4 (261.63 Hz)
    
    Returns
    -------
    AudioSegment
        피치가 변조된 오디오 세그먼트
    """
    if abs(target_hz - base_hz) < 1e-6:
        return seg
    
    # Hz 비율로 factor 계산
    factor = target_hz / base_hz
    
    # 프레임레이트 변경으로 피치 조정
    shifted = seg._spawn(seg.raw_data, overrides={'frame_rate': int(seg.frame_rate * factor)})
    # 다시 원래 SR로 리샘플 → 길이 유지 + 피치만 변조된 효과
    return shifted.set_frame_rate(seg.frame_rate)

# ---- Beep 소리 생성 ----
def generate_beep(frequency_hz: float, duration_ms: int = 200) -> AudioSegment:
    """
    특정 주파수의 Beep 소리 생성
    
    Parameters
    ----------
    frequency_hz : float
        Beep 소리의 주파수 (Hz)
    duration_ms : int
        Beep 소리의 길이 (밀리초)
    
    Returns
    -------
    AudioSegment
        생성된 Beep 소리
    """
    # 사인파 생성
    samples = int(SR * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, samples, False)
    wave = np.sin(2 * np.pi * frequency_hz * t)
    
    # 엔벨로프 적용 (부드러운 시작/끝)
    envelope = np.ones_like(wave)
    fade_samples = int(samples * 0.1)  # 10% fade in/out
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    wave = wave * envelope
    
    # 16-bit PCM으로 변환
    wave = (wave * 32767).astype(np.int16)
    
    # AudioSegment로 변환
    return AudioSegment(
        wave.tobytes(),
        frame_rate=SR,
        sample_width=SAMPLE_WIDTH,
        channels=CHANNELS
    )

# ---- 앞/뒤 침묵 트림 ----
def trim_silence(seg: AudioSegment, threshold_db: int = -45, min_silence_len: int = 60, pad_ms: int = 8) -> AudioSegment:
    if len(seg) == 0:
        return seg
    ns = detect_nonsilent(seg, min_silence_len=min_silence_len, silence_thresh=threshold_db)
    if not ns:
        return seg
    start, end = ns[0][0], ns[-1][1]
    start = max(0, start - pad_ms)
    end = min(len(seg), end + pad_ms)
    return seg[start:end]

# ---- (선택) 속도 조절 (전체 길이 단축) ----
def time_stretch(seg: AudioSegment, rate: float) -> AudioSegment:
    """
    전체 속도 조절 (피치 유지). ffmpeg의 atempo는 0.5~2.0 범위만 허용.
    """
    if abs(rate - 1.0) < 1e-6:
        return seg
    
    # ffmpeg 경로 가져오기 (imageio-ffmpeg 또는 시스템 ffmpeg)
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_path = get_ffmpeg_exe()
    except ImportError:
        ffmpeg_path = "ffmpeg"  # 시스템 ffmpeg 사용
    
    import tempfile, subprocess
    tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        seg.export(tmp_in.name, format="wav")
        # atempo는 0.5~2.0만 지원, 그 외는 단계적 적용 필요
        chain = []
        r = rate
        while r > 2.0:
            chain.append("atempo=2.0")
            r /= 2.0
        while r < 0.5:
            chain.append("atempo=0.5")
            r *= 2.0
        chain.append(f"atempo={r:.6f}")
        filt = ",".join(chain)
        cmd = [ffmpeg_path, "-y", "-i", tmp_in.name, "-filter:a", filt, tmp_out.name]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = AudioSegment.from_file(tmp_out.name, format="wav")
        return _ensure_pcm(out)
    finally:
        try: os.unlink(tmp_in.name)
        except: pass
        try: os.unlink(tmp_out.name)
        except: pass

# ===== 억제 컨텍스트 =====
@dataclass
class SuppressionContext:
    after_eq_topic: bool = False        # Eq의 '은/는' 직후
    after_sum_upto: bool = False        # Sum의 '까지' 직후
    after_prob_pi: bool = False         # Prob의 '확률 피' 직후
    after_prob_bar: bool = False        # Prob의 '바' 직후
    after_trig_power: bool = False      # 삼각함수 제곱 직후 (사인 제곱 엑스 등)

def update_suppression_context(prev_tokens: Sequence[str]) -> SuppressionContext:
    ctx = SuppressionContext()
    if not prev_tokens:
        return ctx
    last = prev_tokens[-1]
    
    # 은/는 직후
    if last in {"은", "는"}: 
        ctx.after_eq_topic = True
    
    # 까지 직후
    if last == "까지": 
        ctx.after_sum_upto = True
    
    # 확률 피 직후
    if len(prev_tokens) >= 2 and prev_tokens[-2:] == ["확률", "피"]:
        ctx.after_prob_pi = True
    
    # 확률...피...바 직후
    if last == "바":
        tail = prev_tokens[-4:]
        if "확률" in tail and "피" in tail:
            ctx.after_prob_bar = True
    
    # 삼각함수 제곱 직후 (사인 제곱, 코사인 제곱, 탄젠트 제곱 등)
    if len(prev_tokens) >= 2:
        trig_functions = {"사인", "코사인", "탄젠트", "시컨트", "코시컨트", "코탄젠트"}
        if prev_tokens[-2] in trig_functions and prev_tokens[-1] == "제곱":
            ctx.after_trig_power = True
    
    return ctx

def should_suppress(ctx: SuppressionContext) -> bool:
    return (ctx.after_eq_topic or ctx.after_sum_upto or 
            ctx.after_prob_pi or ctx.after_prob_bar or ctx.after_trig_power)

# ===== 합성 정책 (개선안) =====
@dataclass
class AudioPolicy:
    # 간격
    base_space_ms: int = 10           # Depth 1 변화 시 기본 간격
    emph_space_ms: int = 500          # Depth 2+ 변화 시 간격
    beep_interval_ms: int = 300       # Beep 소리 간 간격
    
    # 음높이 (Hz) - Depth 변화에 따른 목표 주파수
    # 평범한 음: C4 (261.63 Hz) - 첫 토큰 및 depth 1 변화
    pitch_normal: float = 261.63
    
    # Depth 증가 (깊어짐 = 낮은 음)
    pitch_increase_2: float = 220.0    # 2 증가 음 – 라 A3
    pitch_increase_3: float = 174.61   # 3 증가 음 – 파 F3
    pitch_increase_4: float = 146.83   # 4 증가 음 – 레 D3
    pitch_increase_5: float = 123.47   # 5 증가 음 – 시 B2
    
    # Depth 감소 (얕아짐 = 높은 음)
    pitch_decrease_2: float = 329.63   # 2 감소 음 – 미 E4
    pitch_decrease_3: float = 392.0    # 3 감소 음 – 솔 G4
    pitch_decrease_4: float = 493.88   # 4 감소 음 – 시 B4
    pitch_decrease_5: float = 587.33   # 5 감소 음 – 레 D5
    
    # 전반 속도(말 빠르기)
    speech_rate: float = 1.0

# ===== gTTS 캐시 =====
class GTTSCache:
    def __init__(self, tmpdir: Optional[str] = None):
        self.tmpdir = tmpdir or tempfile.gettempdir()
        # ffmpeg 경로 가져오기
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            self.ffmpeg_path = get_ffmpeg_exe()
        except ImportError:
            self.ffmpeg_path = None

    def _key_mp3(self, text: str) -> str:
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return os.path.join(self.tmpdir, f"gtts_{h}.mp3")

    def tts_to_segment(self, text: str) -> AudioSegment:
        text = (text or "").strip()
        if not text:
            return silence(0)
        path_mp3 = self._key_mp3(text)
        if not os.path.exists(path_mp3):
            gTTS(text=text, lang='ko').save(path_mp3)
        
        # ffmpeg로 직접 변환 (pydub 없이)
        if self.ffmpeg_path:
            import subprocess
            import wave
            import struct
            
            # mp3 -> wav 변환
            path_wav = path_mp3.replace(".mp3", ".wav")
            if not os.path.exists(path_wav):
                cmd = [self.ffmpeg_path, "-y", "-i", path_mp3, "-acodec", "pcm_s16le", 
                       "-ar", str(SR), "-ac", str(CHANNELS), path_wav]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except Exception as e:
                    print(f"ffmpeg 변환 오류: {e}")
                    # fallback
                    seg = AudioSegment.from_file(path_mp3, format="mp3")
                    return _ensure_pcm(seg)
            
            # wav 파일을 직접 읽기
            try:
                with wave.open(path_wav, 'rb') as wf:
                    raw_data = wf.readframes(wf.getnframes())
                    seg = AudioSegment(
                        data=raw_data,
                        sample_width=wf.getsampwidth(),
                        frame_rate=wf.getframerate(),
                        channels=wf.getnchannels()
                    )
                return _ensure_pcm(seg)
            except Exception as e:
                print(f"wav 읽기 오류: {e}")
                # fallback
                seg = AudioSegment.from_file(path_mp3, format="mp3")
                return _ensure_pcm(seg)
        else:
            # ffmpeg 없으면 pydub 사용 (실패할 수 있음)
            seg = AudioSegment.from_file(path_mp3, format="mp3")
            return _ensure_pcm(seg)

# (선택) 발음 보정
def normalize_token_for_ko(token: str) -> str:
    return token

# ===== 합성기 (개선된 알고리즘) =====
class ExpressionAudioReader:
    """
    개선된 소리 알고리즘:
    1. 첫 토큰은 평범한 음(C4)로 낭독
    2. Depth 1 변화: 무시, 평범한 음, 10ms 공백
    3. Depth 2-5 증가: 500ms 공백, 해당 증가 음
    4. Depth 2-5 감소: 500ms 공백, 해당 감소 음
    5. Depth 6+ 변화: Beep 소리 추가 후 나머지는 2-5 단계 음으로
    6. 억제 위치에서는 변화 무시
    """
    def __init__(self, policy: Optional[AudioPolicy] = None, tmpdir: Optional[str] = None):
        self.policy = policy or AudioPolicy()
        self.cache = GTTSCache(tmpdir=tmpdir)

    def _get_target_hz(self, delta: int) -> float:
        """
        Depth 변화량에 따른 목표 Hz 반환
        
        Parameters
        ----------
        delta : int
            Depth 변화량 (양수=증가/깊어짐, 음수=감소/얕아짐)
        
        Returns
        -------
        float
            목표 Hz 값
        """
        abs_delta = abs(delta)
        
        # Depth 6 이상은 5로 처리 (Beep는 별도)
        if abs_delta >= 6:
            abs_delta = (abs_delta % 5) if (abs_delta % 5) != 0 else 5
        
        if delta > 0:  # Depth 증가 (깊어짐 = 낮은 음)
            if abs_delta == 2:
                return self.policy.pitch_increase_2
            elif abs_delta == 3:
                return self.policy.pitch_increase_3
            elif abs_delta == 4:
                return self.policy.pitch_increase_4
            elif abs_delta >= 5:
                return self.policy.pitch_increase_5
        else:  # Depth 감소 (얕아짐 = 높은 음)
            if abs_delta == 2:
                return self.policy.pitch_decrease_2
            elif abs_delta == 3:
                return self.policy.pitch_decrease_3
            elif abs_delta == 4:
                return self.policy.pitch_decrease_4
            elif abs_delta >= 5:
                return self.policy.pitch_decrease_5
        
        return self.policy.pitch_normal

    def synthesize(self, tokens_with_depth: List[Tuple[str, int]]) -> AudioSegment:
        """
        토큰과 depth 정보를 받아 음성 합성
        
        Parameters
        ----------
        tokens_with_depth : List[Tuple[str, int]]
            (토큰, depth) 리스트
        
        Returns
        -------
        AudioSegment
            합성된 오디오
        """
        out = AudioSegment.silent(0, frame_rate=SR)
        prev_depth: Optional[int] = None
        prev_texts: List[str] = []

        for raw_tok, d in tokens_with_depth:
            tok = normalize_token_for_ko(raw_tok)
            
            # 억제 컨텍스트 확인
            sup = should_suppress(update_suppression_context(prev_texts))
            
            # 간격 및 Beep 처리
            if prev_depth is None:
                # 첫 토큰: 기본 간격
                out += silence(self.policy.base_space_ms)
            elif sup:
                # 억제 위치: 기본 간격
                out += silence(self.policy.base_space_ms)
            else:
                delta = d - prev_depth
                abs_delta = abs(delta)
                
                # Depth 1 변화: 무시, 기본 간격
                if abs_delta == 1:
                    out += silence(self.policy.base_space_ms)
                # Depth 2-5 변화: 강조 간격
                elif 2 <= abs_delta <= 5:
                    out += silence(self.policy.emph_space_ms)
                # Depth 6+ 변화: Beep 소리 추가
                elif abs_delta >= 6:
                    # Beep 횟수 계산: abs_delta // 5
                    beep_count = abs_delta // 5
                    
                    # Beep 주파수 선택
                    if delta > 0:  # 증가
                        beep_hz = self.policy.pitch_increase_5  # B2
                    else:  # 감소
                        beep_hz = self.policy.pitch_decrease_5  # D5
                    
                    # Beep 소리 생성 및 추가
                    out += silence(self.policy.emph_space_ms)
                    for i in range(beep_count):
                        if i > 0:
                            out += silence(self.policy.beep_interval_ms)
                        out += generate_beep(beep_hz, duration_ms=200)
                    
                    # Beep 후 간격
                    out += silence(self.policy.beep_interval_ms)

            # 음성 합성
            speech = self.cache.tts_to_segment(tok)
            
            # 전반 속도 조정
            if abs(self.policy.speech_rate - 1.0) > 1e-6:
                speech = time_stretch(speech, self.policy.speech_rate)
            
            # 침묵 제거
            speech = trim_silence(speech, threshold_db=-40, min_silence_len=50, pad_ms=6)

            # 피치 변조
            if prev_depth is None or sup:
                # 첫 토큰 또는 억제 위치: 평범한 음(C4)
                shifted = pitch_shift_to_hz(speech, self.policy.pitch_normal)
            else:
                delta = d - prev_depth
                abs_delta = abs(delta)
                
                # Depth 1 변화: 평범한 음
                if abs_delta == 1:
                    shifted = pitch_shift_to_hz(speech, self.policy.pitch_normal)
                # Depth 2+ 변화: 해당 음높이로 변조
                elif abs_delta >= 2:
                    target_hz = self._get_target_hz(delta)
                    shifted = pitch_shift_to_hz(speech, target_hz)
                else:
                    shifted = speech

            out += shifted
            prev_depth = d
            prev_texts.append(raw_tok)

        return out

    # Colab: 바로 듣기
    def play(self, segment: AudioSegment):
        seg = _ensure_pcm(segment)
        display(Audio(seg.raw_data, rate=seg.frame_rate))

    def save(self, segment: AudioSegment, path: str) -> str:
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

# ===== 편의 함수 =====
def play_tokens(tokens_with_depth: List[Tuple[str, int]],
                policy: Optional[AudioPolicy] = None):
    reader = ExpressionAudioReader(policy=policy)
    seg = reader.synthesize(tokens_with_depth)
    reader.play(seg)
    return reader, seg

def save_tokens(tokens_with_depth: List[Tuple[str, int]],
                out_path: str = "expression.mp3",
                policy: Optional[AudioPolicy] = None) -> str:
    reader = ExpressionAudioReader(policy=policy)
    seg = reader.synthesize(tokens_with_depth)
    return reader.save(seg, out_path)
'''
# ===== 예시 =====
if __name__ == "__main__":
    # 사용 예시
    tokens = [('엑스', 1), ('더하기', 0), ('와이', 2), ('분의', 1), ('일', 2)]
    
    # 재생 (Jupyter 환경에서만 작동)
    # rdr, seg = play_tokens(tokens, policy=AudioPolicy(speech_rate=1.0))
    
    # 파일 저장
    save_tokens(tokens, "expr.mp3", policy=AudioPolicy(speech_rate=1.0))
    print("✓ 오디오 파일 저장 완료: expr.mp3")
'''
