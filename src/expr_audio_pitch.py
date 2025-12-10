# expr_audio_pitch.py
# -*- coding: utf-8 -*-
"""
Expression to Audio Synthesis with Multi-Engine Support

Integrates pitch mapping into AudioPolicy and supports both gTTS and edge-tts engines.
Implements depth_change and grouping_pitch algorithms with suppression context handling.

Features:
- Dual TTS engines: gTTS (male voice) and edge-tts (female voice)
- Integrated pitch/speed mapping in AudioPolicy
- Rubberband pitch shifting for high quality
- Suppression context for natural-sounding depth changes
- Two main functions: latex_audio_depth_change() and latex_audio_grouping_pitch()
"""

import os
import time
import hashlib
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Literal
import asyncio

import numpy as np
import torch
import torchaudio.functional as F
import soundfile as sf
from gtts import gTTS

try:
    from IPython.display import Audio, display
except ImportError:
    Audio = None
    display = print

# Import existing modules
from LaTeX_Parser import latex_to_expression  
from Expression_Syntax import expression_to_korean_with_depth, expression_to_tokens_with_pitch


# ===== Constants =====
SR = 22050  # Sample rate
CHANNELS = 1


# ===== Audio Utilities =====

def change_speed(waveform: torch.Tensor, sample_rate: int, speed_factor: float) -> torch.Tensor:
    """Change speed without changing pitch using Phase Vocoder."""
    if abs(speed_factor - 1.0) < 1e-6:
        return waveform
    
    n_fft = 1024
    hop_length = 256
    
    # STFT
    spec = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, 
                      return_complex=True, window=torch.hann_window(n_fft))
    
    # Time Stretch
    import torchaudio.transforms as T
    stretcher = T.TimeStretch(hop_length=hop_length, n_freq=n_fft//2 + 1, 
                               fixed_rate=speed_factor)
    stretched_spec = stretcher(spec)
    
    # ISTFT
    new_len = int(waveform.shape[-1] / speed_factor)
    new_waveform = torch.istft(stretched_spec, n_fft=n_fft, hop_length=hop_length,
                                length=new_len, window=torch.hann_window(n_fft))
    
    return new_waveform


def silence(dur_ms: int) -> torch.Tensor:
    """Generate silence of specified duration."""
    samples = int(SR * dur_ms / 1000)
    return torch.zeros(1, samples)


def generate_beep(frequency_hz: float, duration_ms: int = 200) -> torch.Tensor:
    """Generate a beep tone with fade in/out envelope."""
    samples = int(SR * duration_ms / 1000)
    t = torch.linspace(0, duration_ms / 1000, samples)
    wave = torch.sin(2 * np.pi * frequency_hz * t)
    
    # Envelope
    envelope = torch.ones_like(wave)
    fade_samples = int(samples * 0.1)
    if fade_samples > 0:
        fade_in = torch.linspace(0, 1, fade_samples)
        fade_out = torch.linspace(1, 0, fade_samples)
        envelope[:fade_samples] = fade_in
        envelope[-fade_samples:] = fade_out
    
    wave = wave * envelope
    return wave.unsqueeze(0)


def trim_silence_torch(waveform: torch.Tensor, threshold_db: float = -40, 
                       pad_ms: int = 6) -> torch.Tensor:
    """Trim silence from start and end of waveform."""
    if waveform.shape[1] == 0:
        return waveform
    
    threshold_linear = 10 ** (threshold_db / 20)
    amp = waveform.abs()
    mask = amp > threshold_linear
    mask = mask.any(dim=0)
    
    indices = torch.nonzero(mask)
    if indices.numel() == 0:
        return torch.zeros(waveform.shape[0], 0)
    
    start = indices[0].item()
    end = indices[-1].item()
    
    pad_samples = int(SR * pad_ms / 1000)
    start = max(0, start - pad_samples)
    end = min(waveform.shape[1], end + pad_samples)
    
    return waveform[:, start:end]


def soft_limiter(waveform: torch.Tensor, gain: float = 1.0, threshold: float = 0.9) -> torch.Tensor:
    """
    Apply soft limiting to increase volume naturally without harsh clipping.
    Uses tanh for soft clipping above threshold.
    
    Args:
        waveform: Input audio tensor
        gain: Volume gain factor
        threshold: Threshold for soft clipping start (0.0 to 1.0)
        
    Returns:
        Processed waveform
    """
    if abs(gain - 1.0) < 1e-6:
        return waveform
        
    # Apply gain
    amplified = waveform * gain
    
    # Soft clipping using tanh
    # x < threshold: linear
    # x >= threshold: tanh compression
    
    # Simple tanh approach for smooth limiting
    # output = tanh(input)
    # But we want linear behavior for low amplitudes.
    
    # Let's use a simple tanh soft clipper:
    # y = tanh(x)
    
    return torch.tanh(amplified)


def apply_rubberband_pitch(waveform: torch.Tensor, sample_rate: int, 
                           semitones: float) -> torch.Tensor:
    """
    Apply pitch shift using ffmpeg rubberband filter.
    Falls back to torchaudio if rubberband is not available.
    
    Args:
        waveform: Input audio (1, T)
        sample_rate: Sample rate
        semitones: Pitch shift in semitones
    
    Returns:
        Pitch-shifted waveform
    """
    if abs(semitones) < 1e-6:
        return waveform
    
    try:
        # Check if ffmpeg is available
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError("ffmpeg not found")
        
        # Create temp files
        temp_input = tempfile.mktemp(suffix='.wav')
        temp_output = tempfile.mktemp(suffix='.wav')
        
        # Save input
        wav_np = waveform[0].numpy() if waveform.ndim > 1 else waveform.numpy()
        sf.write(temp_input, wav_np, sample_rate)
        
        # Calculate pitch ratio
        pitch_ratio = 2 ** (semitones / 12)
        
        # Run ffmpeg with rubberband
        cmd = [
            'ffmpeg', '-y', '-i', temp_input,
            '-filter:a', f'rubberband=pitch={pitch_ratio}',
            temp_output
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Load processed audio
            wav_shifted, _ = sf.read(temp_output)
            waveform = torch.from_numpy(wav_shifted).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
        else:
            # Fallback to torchaudio
            waveform = F.pitch_shift(waveform, sample_rate, n_steps=semitones)
        
        # Clean up
        try:
            os.remove(temp_input)
            os.remove(temp_output)
        except:
            pass
    
    except Exception:
        # Fallback to torchaudio pitch shift
        waveform = F.pitch_shift(waveform, sample_rate, n_steps=semitones)
    
    return waveform


# ===== Suppression Context Detection =====

def should_suppress_depth_change(prev_tokens: List[str]) -> bool:
    """
    Check if current position is after a suppression context token.
    
    Suppression contexts (from Expression_Syntax.py analysis):
    1. Eq: "은" or "는"
    2. Sum: "까지"  
    3. Prob: "확률 피" or "바"
    4. Bar: "바"
    5. Power(trig): "사인/코사인/... 제곱"
    
    Args:
        prev_tokens: List of previous token strings
    
    Returns:
        True if depth change effects should be suppressed
    """
    if not prev_tokens:
        return False
    
    last = prev_tokens[-1]
    
    # Direct suppression tokens
    if last in ["은", "는", "까지"]:
        return True
    
    # "바" token (from Prob or Bar)
    if last == "바":
        return True
    
    # "확률 피" pattern
    if last == "피" and len(prev_tokens) >= 2 and "확률" in prev_tokens[-3:]:
        return True
    
    # "제곱" after trigonometric functions
    if last == "제곱" and len(prev_tokens) >= 2:
        trig_names = ["사인", "코사인", "탄젠트", "시컨트", "코시컨트", "코탄젠트"]
        if prev_tokens[-2] in trig_names:
            return True
    
    return False


# ===== Enhanced AudioPolicy with Integrated Pitch Mapping =====

@dataclass
class AudioPolicy:
    """
    Enhanced audio synthesis policy with integrated pitch/speed mapping.
    
    Parameters
    ----------
    is_male : bool
        If True, use gTTS (male voice). If False, use edge-tts (female voice).
    voice : str
        Voice ID for edge-tts (ignored for gTTS).
    pitch_mapping : Dict[int, float]
        Mapping from pitch level to semitone shift. If None, uses default.
    speed_mapping : Dict[int, float]
        Mapping from pitch level to speed factor. If None, uses default (1.0 for all).
    pitch_step : float
        Semitones per depth level for depth_change algorithm.
    base_space_ms : int
        Base spacing between tokens (milliseconds).
    emph_space_ms : int
        Emphasized spacing for depth changes of 2-5 levels.
    beep_interval_ms : int
        Interval between beeps for large depth jumps.
    base_volume : float
        Base volume multiplier.
    base_speed : float
        Base speed multiplier.
    use_rubberband : bool
        If True, use ffmpeg rubberband for pitch shifting (higher quality).
    """
    # Voice selection
    is_male: bool = False  # False = edge-tts (female), True = gTTS (male)
    voice: str = "ko-KR-SunHiNeural"  # Edge-TTS voice
    
    # Integrated pitch mapping (replaces pitch_mapping.py)
    pitch_mapping: Optional[Dict[int, float]] = None
    speed_mapping: Optional[Dict[int, float]] = None
    
    # Depth-change algorithm parameters
    pitch_step: float = -0.5  # Semitones per depth level
    
    # Spacing configuration
    base_space_ms: int = 10
    emph_space_ms: int = 100
    beep_interval_ms: int = 100
    
    # Base settings
    base_volume: float = 5.0  # edge-tts 볼륨이 작아서 기본값 5.0으로 설정
    base_speed: float = 1
    
    # Audio processing
    use_rubberband: bool = True
    target_lufs: float = -14.0  # Target LUFS for normalization
    
    def __post_init__(self):
        """Initialize default mappings if not provided."""
        if self.pitch_mapping is None:
            # Default pitch mapping (부호 반전: 양수 level = 낮은 음)
            # Level이 증가할수록 pitch가 낮아짐 (음수 semitones)
            self.pitch_mapping = {
                -4: 12, 
                -3: 10, 
                -2: 8, 
                -1: 6, 
                0: 4,
                1: 2, 
                2: 1, 
                3: 0, 
                4: -1,
                5: -3, 
                6: -5, 
                7: -7, 
                8: -9, 
                9: -11, 
                10: -13
            }
        
        if self.speed_mapping is None:
            # Default: constant speed
            self.speed_mapping = {i: 1.0 for i in range(-11, 11)}
    
    def level_to_semitones(self, level: int) -> float:
        """Convert pitch level to semitone shift via mapping."""
        return self._interpolate(level, self.pitch_mapping)
    
    def level_to_speed(self, level: int) -> float:
        """Convert pitch level to speed factor via mapping."""
        return self._interpolate(level, self.speed_mapping)
    
    def _interpolate(self, level, mapping):
        """Linear interpolation for pitch/speed mapping."""
        if level in mapping:
            return mapping[level]
        
        levels = sorted(mapping.keys())
        
        # Extrapolate beyond range
        if level < levels[0]:
            step = mapping[levels[0]] - mapping[levels[1]]
            return mapping[levels[0]] + step * (levels[0] - level)
        elif level > levels[-1]:
            step = mapping[levels[-2]] - mapping[levels[-1]]
            return mapping[levels[-1]] - step * (level - levels[-1])
        
        # Interpolate between two points
        for i in range(len(levels) - 1):
            if levels[i] <= level <= levels[i + 1]:
                l1, l2 = levels[i], levels[i + 1]
                v1, v2 = mapping[l1], mapping[l2]
                ratio = (level - l1) / (l2 - l1)
                return v1 + ratio * (v2 - v1)
        
        return 0.0  # Fallback


def apply_lufs_normalization(waveform: torch.Tensor, sample_rate: int, target_lufs: float = -14.0) -> torch.Tensor:
    """
    Normalize audio to target LUFS using pyloudnorm.
    
    Args:
        waveform: Input audio tensor (1, T) or (C, T)
        sample_rate: Sample rate
        target_lufs: Target integrated loudness in LUFS
        
    Returns:
        Normalized waveform
    """
    try:
        import pyloudnorm as pyln
    except ImportError:
        print("pyloudnorm not found. Skipping LUFS normalization. Install with: pip install pyloudnorm")
        return waveform

    # Convert to numpy (T, C) for pyloudnorm
    if waveform.ndim == 1:
        wav_np = waveform.numpy()
    else:
        wav_np = waveform.t().numpy()
    
    # Measure loudness
    try:
        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(wav_np)
        
        # Normalize
        normalized_audio = pyln.normalize.loudness(wav_np, loudness, target_lufs)
        
        # Convert back to tensor
        waveform_norm = torch.from_numpy(normalized_audio).float()
        if waveform.ndim == 1:
            # If input was 1D, output should be 1D? pyloudnorm might return 1D if input was 1D
            pass 
        elif waveform.ndim == 2 and waveform_norm.ndim == 2:
             waveform_norm = waveform_norm.t() # (T, C) -> (C, T)
             
        # Handle shape mismatch if any (pyloudnorm usually preserves shape)
        if waveform.ndim == 2 and waveform_norm.ndim == 1:
            waveform_norm = waveform_norm.unsqueeze(0)
            
        return waveform_norm
        
    except Exception as e:
        print(f"LUFS normalization failed: {e}")
        return waveform


# ===== TTS Engine Classes =====

class GTTSCache:
    """
    Cache for gTTS-generated audio files.
    
    Stores audio in temporary directory with hash-based filenames.
    """
    def __init__(self, tmpdir: Optional[str] = None):
        self.tmpdir = tmpdir or tempfile.gettempdir()
        self.cache_dir = os.path.join(self.tmpdir, "gtts_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_hash(self, text: str) -> str:
        """Get hash for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def tts_to_tensor(self, text: str) -> torch.Tensor:
        """
        Generate TTS audio and return as tensor.
        
        Args:
            text: Korean text to synthesize
        
        Returns:
            Audio waveform tensor (1, T)
        """
        text_hash = self._get_hash(text)
        cache_file = os.path.join(self.cache_dir, f"{text_hash}.mp3")
        
        # Generate if not cached
        if not os.path.exists(cache_file):
            try:
                tts = gTTS(text=text, lang='ko')
                tts.save(cache_file)
            except Exception as e:
                print(f"gTTS error for '{text}': {e}")
                # Return short silence on error
                return silence(100)
        
        # Load audio
        try:
            wav, sr = sf.read(cache_file)
            waveform = torch.from_numpy(wav).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()  # (T, C) -> (C, T)
            
            # Resample if needed
            if sr != SR:
                import torchaudio.transforms as T
                resamp = T.Resample(sr, SR)
                waveform = resamp(waveform)
            
            return waveform
        except Exception as e:
            print(f"Failed to load cached audio for '{text}': {e}")
            return silence(100)


class EdgeTTSEngine:
    """Engine for edge-tts synthesis."""
    
    def __init__(self, voice: str = "ko-KR-SunHiNeural"):
        self.voice = voice
        
        # Import and setup edge-tts
        try:
            import edge_tts
            import nest_asyncio
            try:
                nest_asyncio.apply()
            except:
                pass
            self.edge_tts = edge_tts
        except ImportError:
            raise ImportError(
                "edge-tts not installed. Install with: pip install edge-tts nest-asyncio"
            )
    
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for edge-tts synthesis.
        
        - Removes extra whitespace
        - Validates minimum length
        """
        import re
        
        # Remove leading/trailing whitespace
        normalized = text.strip()
        
        # Normalize multiple spaces to single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def synthesize_to_tensor(self, text: str) -> torch.Tensor:
        """
        Synthesize text to audio tensor using edge-tts.
        
        Shows warnings when synthesis fails but does NOT fallback to gTTS.
        
        Args:
            text: Korean text to synthesize
        
        Returns:
            Audio waveform tensor (1, T), or silence if failed
        """
        # Normalize text
        normalized_text = self._normalize_text(text)
        
        # Handle empty text
        if not normalized_text:
            print(f"⚠️  edge-tts warning: Empty text after normalization, returning silence")
            return silence(100)
        
        # Check minimum length (without spaces)
        text_no_spaces = normalized_text.replace(" ", "")

        temp_file = tempfile.mktemp(suffix='.mp3')
        
        async def _generate():
            communicate = self.edge_tts.Communicate(
                normalized_text,
                self.voice,
                rate="+0%",  # Neutral rate
                pitch="+0Hz",  # Neutral pitch
                volume="+200%"  # Increased volume
            )
            await communicate.save(temp_file)
        
        try:
            asyncio.run(_generate())
            
            # Validate file was created and has content
            if not os.path.exists(temp_file):
                raise RuntimeError(f"No audio file created for '{text}'")
            
            file_size = os.path.getsize(temp_file)
            if file_size == 0:
                raise RuntimeError(f"No audio was received (empty file)")
            
            # Load audio
            wav_np, sample_rate = sf.read(temp_file)
            waveform = torch.from_numpy(wav_np).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()
            
            # Resample if needed
            if sample_rate != SR:
                import torchaudio.transforms as T
                resamp = T.Resample(sample_rate, SR)
                waveform = resamp(waveform)
            
            # Clean up
            try:
                os.remove(temp_file)
            except:
                pass
            
            return waveform
        
        except Exception as e:
            print(f"⚠️  edge-tts error for '{text}': {e}")
            
            # Clean up temp file if it exists
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            
            # Return silence (no fallback to gTTS)
            return silence(100)




# ===== Main Synthesis Functions =====

def latex_audio_depth_change(
    latex: str,
    output_dir: str = "latex_to_audio/single_example",
    filename: str = "single_test.mp3",
    is_male: bool = False,
    is_naive: bool = False,
    policy: Optional[AudioPolicy] = None
) -> str:
    """
    Generate audio from LaTeX using depth_change algorithm.
    
    The depth_change algorithm responds to changes in depth, not absolute depth.
    Pitch changes, spacing, and beeps are applied based on depth deltas.
    
    Suppression contexts (no auditory stimuli despite depth changes):
    - After Eq: "은/는"
    - After Sum: "까지"
    - After Prob: "확률 피" or "바"
    - After Bar: "바"
    - After trig Power: "사인/코사인/... 제곱"
    
    Args:
        latex: LaTeX mathematical expression
        output_dir: Output directory (relative path)
        filename: Output filename
        is_male: If True, use gTTS (male). If False, use edge-tts (female)
        is_naive: If True, use practical reading style (passed to expression_to_korean)
        policy: AudioPolicy for configuration. If None, uses default.
    
    Returns:
        Absolute path to generated audio file
    """
    # Create policy if not provided
    if policy is None:
        if is_male:
            policy = AudioPolicy(is_male=is_male, base_volume=3, base_speed=1.5)
        else:
            policy = AudioPolicy(is_male=is_male, base_volume=5, base_speed=1.2)
    
    # Parse LaTeX to Expression
    from LaTeX_Parser import latex_to_expression
    from Expression_Syntax import expression_to_korean
    expr = latex_to_expression(latex)
    
    # Get tokens with depth
    tokens_with_depth = expression_to_korean_with_depth(expr, depth=0, is_naive=is_naive)
    
    # Print token analysis
    print("=" * 70)
    print(f"LaTeX 수식: {latex}")
    print(f"알고리즘: depth_change")
    print(f"is_naive: {is_naive}")
    print("=" * 70)
    
    # Print alternative text
    alt_text_false = expression_to_korean(expr, is_naive=False)
    alt_text_true = expression_to_korean(expr, is_naive=True)
    print(f"\n대체 텍스트 (is_naive=False): {alt_text_false}")
    print(f"대체 텍스트 (is_naive=True):  {alt_text_true}")
    
    # Print tokens with depth
    print(f"\n토큰 개수: {len(tokens_with_depth)}")
    print("\n토큰 + Depth:")
    print(f"{'번호':<6} {'토큰':<20} {'Depth':<8} {'Delta':<8}")
    print("-" * 50)
    for i, (token, depth) in enumerate(tokens_with_depth):
        if i == 0:
            delta_str = "-"
        else:
            delta = depth - tokens_with_depth[i-1][1]
            delta_str = f"{delta:+d}"
        print(f"{i+1:<6} {token:<20} {depth:<8} {delta_str:<8}")
    print("-" * 70 + "\n")
    
    # Initialize TTS engine
    if policy.is_male:
        tts_cache = GTTSCache()
        def synthesize(text):
            return tts_cache.tts_to_tensor(text)
    else:
        tts_engine = EdgeTTSEngine(voice=policy.voice)
        def synthesize(text):
            return tts_engine.synthesize_to_tensor(text)
    
    # Process tokens with depth_change algorithm
    segments = []
    prev_depth = None
    prev_tokens = []
    
    # Buffer for batching tokens with small delta
    pending_tokens = []
    
    for i, (token, depth) in enumerate(tokens_with_depth):
        # Calculate depth delta
        if prev_depth is not None:
            delta = depth - prev_depth
            abs_delta = abs(delta)
        else:
            delta = 0
            abs_delta = 0
            
        # Check suppression context
        is_suppressed = should_suppress_depth_change(prev_tokens)
        
        # Determine if we should batch this token
        # Batch if small delta OR suppressed (suppressed tokens have no pitch shift/beeps)
        if abs_delta <= 1 or is_suppressed:
            pending_tokens.append(token)
        else:
            # Large jump AND not suppressed
            
            # 1. Flush pending tokens first
            if pending_tokens:
                combined_text = " ".join(pending_tokens)
                base_audio = synthesize(combined_text)
                base_audio = trim_silence_torch(base_audio)
                
                # Apply speed change (base speed only)
                if abs(policy.base_speed - 1.0) > 1e-6:
                    base_audio = change_speed(base_audio, SR, policy.base_speed)
                
                # Apply volume using soft limiter
                if abs(policy.base_volume - 1.0) > 1e-6:
                    base_audio = soft_limiter(base_audio, policy.base_volume)
                
                segments.append(base_audio)
                pending_tokens = []
            
            # 2. Process current token (Large Delta)
            # Calculate beeps and spacing
            if 2 <= abs_delta <= 5:
                segments.append(silence(policy.emph_space_ms))
                semitones = policy.level_to_semitones(delta)
            else: # abs_delta >= 6
                beep_count = abs_delta // 5
                remaining = abs_delta % 5
                
                beep_freq = 1000 if delta > 0 else 800
                for _ in range(beep_count):
                    segments.append(generate_beep(beep_freq, duration_ms=200))
                    segments.append(silence(policy.beep_interval_ms))
                
                segments.append(silence(policy.emph_space_ms))
                
                if remaining == 0:
                    semitones = 0
                else:
                    sign = 1 if delta > 0 else -1
                    semitones = policy.level_to_semitones(sign * remaining)
            
            # Synthesize current token
            token_audio = synthesize(token)
            token_audio = trim_silence_torch(token_audio)
            
            if abs(policy.base_speed - 1.0) > 1e-6:
                token_audio = change_speed(token_audio, SR, policy.base_speed)
            
            if policy.use_rubberband:
                token_audio = apply_rubberband_pitch(token_audio, SR, semitones)
            else:
                token_audio = F.pitch_shift(token_audio, SR, n_steps=semitones)
            
            if abs(policy.base_volume - 1.0) > 1e-6:
                token_audio = soft_limiter(token_audio, policy.base_volume)
            
            segments.append(token_audio)
        
        # Update tracking
        prev_depth = depth
        prev_tokens.append(token)
        
    # Flush remaining pending tokens
    if pending_tokens:
        combined_text = " ".join(pending_tokens)
        base_audio = synthesize(combined_text)
        base_audio = trim_silence_torch(base_audio)
        
        if abs(policy.base_speed - 1.0) > 1e-6:
            base_audio = change_speed(base_audio, SR, policy.base_speed)
        
        if abs(policy.base_volume - 1.0) > 1e-6:
            base_audio = soft_limiter(base_audio, policy.base_volume)
        
        segments.append(base_audio)
    
    # Concatenate all segments
    final_audio = torch.cat(segments, dim=1)
    
    # Apply LUFS normalization
    final_audio = apply_lufs_normalization(final_audio, SR, policy.target_lufs)
    
    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Save as wav first, then convert if needed
    temp_wav = tempfile.mktemp(suffix='.wav')
    wav_np = final_audio[0].numpy()
    sf.write(temp_wav, wav_np, SR)
    
    if filename.endswith('.mp3'):
        # Convert to mp3 using ffmpeg
        try:
            subprocess.run(['ffmpeg', '-y', '-i', temp_wav, '-codec:a', 'libmp3lame', output_path],
                          capture_output=True, check=True)
            os.remove(temp_wav)
        except:
            # Fallback: save as wav
            os.rename(temp_wav, output_path.replace('.mp3', '.wav'))
            output_path = output_path.replace('.mp3', '.wav')
    else:
        os.rename(temp_wav, output_path)
    
    # Display audio
    if Audio is not None:
        display(Audio(output_path))
    
    return os.path.abspath(output_path)


def latex_audio_grouping_pitch(
    latex: str,
    output_dir: str = "latex_to_audio/single_example",
    filename: str = "single_test.mp3",
    is_male: bool = False,
    is_naive: bool = False,
    policy: Optional[AudioPolicy] = None
) -> str:
    """
    Generate audio from LaTeX using grouping_pitch algorithm.
    
    The grouping_pitch algorithm assigns pitch and volume based on structural grouping.
    Each token gets a pitch_level and volume_level from expression_to_tokens_with_pitch().
    
    Args:
        latex: LaTeX mathematical expression
        output_dir: Output directory (relative path)
        filename: Output filename
        is_male: If True, use gTTS (male). If False, use edge-tts (female)
        is_naive: If True, use practical reading style (passed to expression_to_tokens)
        policy: AudioPolicy for configuration. If None, uses default.
    
    Returns:
        Absolute path to generated audio file
    """
    # Create policy if not provided
    if policy is None:
        if is_male:
            policy = AudioPolicy(is_male=is_male, base_volume=3, base_speed=1.5)
        else:
            policy = AudioPolicy(is_male=is_male, base_volume=5, base_speed=1.2)
    
    # Parse LaTeX to Expression
    from LaTeX_Parser import latex_to_expression
    from Expression_Syntax import expression_to_korean
    expr = latex_to_expression(latex)
    
    # Get tokens with pitch and volume levels
    tokens_with_pitch_vol = expression_to_tokens_with_pitch(expr, d=0, is_naive=is_naive)
    
    print(tokens_with_pitch_vol)

    # Print token analysis
    print("=" * 70)
    print(f"LaTeX 수식: {latex}")
    print(f"알고리즘: grouping_pitch")
    print(f"is_naive: {is_naive}")
    print("=" * 70)
    
    # Print alternative text
    alt_text_false = expression_to_korean(expr, is_naive=False)
    alt_text_true = expression_to_korean(expr, is_naive=True)
    print(f"\n대체 텍스트 (is_naive=False): {alt_text_false}")
    print(f"대체 텍스트 (is_naive=True):  {alt_text_true}")
    
    # Print tokens with pitch and volume
    print(f"\n토큰 개수: {len(tokens_with_pitch_vol)}")
    print("\n토큰 + Pitch Level + Volume Level:")
    print(f"{'번호':<6} {'토큰':<20} {'Pitch Lv':<10} {'Volume Lv':<10}")
    print("-" * 50)
    for i, (token, pitch_lv, vol_lv) in enumerate(tokens_with_pitch_vol):
        print(f"{i+1:<6} {token:<20} {pitch_lv:<10} {vol_lv:<10}")
    print("-" * 70 + "\n")
    
    # Initialize TTS engine
    if policy.is_male:
        tts_cache = GTTSCache()
        def synthesize(text):
            return tts_cache.tts_to_tensor(text)
    else:
        tts_engine = EdgeTTSEngine(voice=policy.voice)
        def synthesize(text):
            return tts_engine.synthesize_to_tensor(text)
    
    # Process tokens with grouping_pitch algorithm
    segments = []
    
    # Batch processing variables
    pending_tokens = []
    current_pitch_level = None
    current_volume_level = None
    
    for i, (token, pitch_level, volume_level) in enumerate(tokens_with_pitch_vol):
        # Check if we can batch this token
        if current_pitch_level is None:
            # First token
            current_pitch_level = pitch_level
            current_volume_level = volume_level
            pending_tokens.append(token)
        elif pitch_level == current_pitch_level and volume_level == current_volume_level:
            # Same levels, add to batch
            pending_tokens.append(token)
        else:
            # Different levels, flush pending batch
            combined_text = " ".join(pending_tokens)
            
            # Add spacing before batch (if not first batch)
            if len(segments) > 0:
                segments.append(silence(policy.base_space_ms))
            
            # Synthesize batch
            base_audio = synthesize(combined_text)
            base_audio = trim_silence_torch(base_audio)
            
            # Apply effects using current_pitch_level and current_volume_level
            semitones = policy.level_to_semitones(current_pitch_level)
            speed_factor = policy.level_to_speed(current_pitch_level) * policy.base_speed
            
            if abs(speed_factor - 1.0) > 1e-6:
                base_audio = change_speed(base_audio, SR, speed_factor)
            
            if policy.use_rubberband:
                audio = apply_rubberband_pitch(base_audio, SR, semitones)
            else:
                audio = F.pitch_shift(base_audio, SR, n_steps=semitones)
            
            # Volume adjustment
            if current_volume_level >= 1:
                level_factor = current_volume_level + 10
            elif current_volume_level <= -1:
                level_factor = 0.7 * 1 / (-current_volume_level)
            else:
                level_factor = 1.0
            
            volume_factor = policy.base_volume * level_factor
            audio = soft_limiter(audio, volume_factor)
            
            segments.append(audio)
            
            # Start new batch
            pending_tokens = [token]
            current_pitch_level = pitch_level
            current_volume_level = volume_level
            
    # Flush remaining pending tokens
    if pending_tokens:
        combined_text = " ".join(pending_tokens)
        
        if len(segments) > 0:
            segments.append(silence(policy.base_space_ms))
            
        base_audio = synthesize(combined_text)
        base_audio = trim_silence_torch(base_audio)
        
        semitones = policy.level_to_semitones(current_pitch_level)
        speed_factor = policy.level_to_speed(current_pitch_level) * policy.base_speed
        
        if abs(speed_factor - 1.0) > 1e-6:
            base_audio = change_speed(base_audio, SR, speed_factor)
        
        if policy.use_rubberband:
            audio = apply_rubberband_pitch(base_audio, SR, semitones)
        else:
            audio = F.pitch_shift(base_audio, SR, n_steps=semitones)
        
        if current_volume_level >= 1:
            level_factor = current_volume_level + 0.5
        elif current_volume_level <= -1:
            level_factor = 0.7 * 1 / (-current_volume_level)
        else:
            level_factor = 1.0
        
        volume_factor = policy.base_volume * level_factor
        audio = soft_limiter(audio, volume_factor)
        
        segments.append(audio)
    
    # Concatenate all segments
    final_audio = torch.cat(segments, dim=1)
    
    # Apply LUFS normalization
    final_audio = apply_lufs_normalization(final_audio, SR, policy.target_lufs)
    
    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Save as wav first, then convert if needed
    temp_wav = tempfile.mktemp(suffix='.wav')
    wav_np = final_audio[0].numpy()
    sf.write(temp_wav, wav_np, SR)
    
    if filename.endswith('.mp3'):
        # Convert to mp3 using ffmpeg
        try:
            subprocess.run(['ffmpeg', '-y', '-i', temp_wav, '-codec:a', 'libmp3lame', output_path],
                          capture_output=True, check=True)
            os.remove(temp_wav)
        except:
            # Fallback: save as wav
            os.rename(temp_wav, output_path.replace('.mp3', '.wav'))
            output_path = output_path.replace('.mp3', '.wav')
    else:
        os.rename(temp_wav, output_path)
    
    # Display audio
    if Audio is not None:
        display(Audio(output_path))
    
    return os.path.abspath(output_path)


# ===== Convenience Functions =====

def create_custom_policy(
    is_male: bool = False,
    pitch_mapping: Optional[Dict[int, float]] = None,
    speed_mapping: Optional[Dict[int, float]] = None,
    pitch_step: float = -0.5,
    **kwargs
) -> AudioPolicy:
    """
    Create a custom AudioPolicy with specified parameters.
    
    Args:
        is_male: Voice selection (True=gTTS male, False=edge-tts female)
        pitch_mapping: Custom pitch level → semitones mapping
        speed_mapping: Custom pitch level → speed mapping
        pitch_step: Semitones per depth level for depth_change
        **kwargs: Additional AudioPolicy parameters
    
    Returns:
        Configured AudioPolicy instance
    """
    return AudioPolicy(
        is_male=is_male,
        pitch_mapping=pitch_mapping,
        speed_mapping=speed_mapping,
        pitch_step=pitch_step,
        **kwargs
    )

