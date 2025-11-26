from LaTeX_Parser import *
from speech_synthesizer import *
from Expression_Syntax import *
from gtts_expr_audio_pitch import *
from audio_pitch import *
from grouping_pitch import *



def latex_audio_grouping_pitch(latex, output_dir="latex_to_audio/single_example", filename="single_test.mp3", auto_play=True):

    expression = latex_to_expression(latex)
    korean_text = expression_to_korean(expression)
    tokens_pitch_volume = expr_to_tokens_with_pitch(expression)

    print("LaTeX:", latex)
    print("수식:", expression)
    print("Expression 객체:", repr(expression))
    print("한국어:", korean_text)

    token_display = [f"{tok}({lvl:+d})" for tok, lvl, _ in tokens_pitch_volume]
    print("토큰:", " ".join(token_display))

    print("\n→ 음성 합성 중...", end=" ")
    audio = synthesizer.synthesize(tokens_pitch_volume)
    duration = len(audio) / 1000
    print(f"✓ {duration:.2f}초")
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)
    synthesizer.save(audio, filepath)
    print("→ 저장:", filepath)

    print("\nPitch Level 분석:")
    for token, level, volume in tokens_pitch_volume:
        hz = mapper.get_hz(level)
        note = mapper.get_note_name(level)
        vol = f", volume {volume:+d}" if volume else ""
        print(f"  '{token}' → Level {level:+2d} ({note}, {hz:.2f} Hz{vol})")

    display(Audio(filename=filepath, autoplay=auto_play))
