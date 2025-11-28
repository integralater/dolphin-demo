import streamlit as st
import os
import tempfile
from gtts import gTTS

# 가이드라인에 명시된 핵심 모듈 임포트
from LaTeX_Parser import latex_to_expression
from Expression_Syntax import expression_to_korean, expression_to_tokens_with_pitch
from speech_synthesizer import MathSpeechSynthesizer
from gtts_expr_audio_pitch import AudioPolicy
from grouping_pitch import latex_audio_grouping_pitch

# ----------------- A. 페이지 설정 -----------------
st.set_page_config(
    page_title="Dolphin Math TTS",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ----------------- B. 사이드바 옵션 설정 -----------------
st.sidebar.title("🎛️ 옵션 설정")

# 1. 발음 스타일 선택
style_option = st.sidebar.selectbox(
    "발음 스타일 (Style)",
    ("non-pitch change", "depth version", "grouping version"),
    index=2, # 기본값: Expressive
    help="non-pitch changle: 높낮이 없음\nepth version: d 자연스러운 피치\nHierarchical: 구조 강조형"
)

# 2. 구어체 모드 선택
is_naive = st.sidebar.checkbox(
    "구어체 모드 (Casual)",
    value=True,
    help="체크 시: '이 분의 일' (자연스러움)\n해제 시: 형식적인 수학 표현"
)

st.sidebar.markdown("---")
st.sidebar.info("Dolphin-doing-Math Project\nLatex to Korean Speech")

# ----------------- C. 메인 화면 구성 -----------------
st.title("🔢 LaTeX 수식 음성 합성 데모")
#st.markdown(f"현재 설정: **{style_option}** 스타일 | **{'구어체' if is_naive else '형식적'}** 모드")

# 입력창
latex_input = st.text_area(
    "LaTeX 수식을 입력하세요:",
    value=r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}",
    height=120
)

# ----------------- D. 실시간 분석 및 변환 로직 -----------------
if latex_input.strip():
    col1, col2 = st.columns(2)
    
    # [왼쪽 컬럼] 수식 렌더링
    with col1:
        st.subheader("수식 미리보기")
        st.latex(latex_input)

    # 파싱 및 텍스트 변환 시도
    try:
        # 1. LaTeX 파싱 (핵심 함수 1)
        expr = latex_to_expression(latex_input)
        
        # 2. 한국어 텍스트 변환 (핵심 함수 2)
        korean_text = expression_to_korean(expr, is_naive = is_naive)
        
        # [오른쪽 컬럼] 변환된 한국어 텍스트 표시 (사용자 경험 개선)
        with col2:
            st.subheader("한국어 발음 텍스트")
            st.info(korean_text)
            
        # 내부 구조 디버깅용 (필요 시 확장)
        with st.expander("개발자용: 내부 AST 구조 확인"):
            st.text(repr(expr))

    except Exception as e:
        st.error(f"LaTeX 파싱 오류: {e}")
        st.stop() # 오류 발생 시 아래 로직 중단

    st.markdown("---")

    # ----------------- E. 음성 변환 및 재생 버튼 -----------------
    if st.button("🔊 음성 변환 및 재생", type="primary"):
        with st.spinner(f"=음성을 생성 중입니다..."):
            
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                output_path = tmp_file.name

            try:# 스타일별 분기 처리 (가이드라인 '음원 생성 방법' 참조)
                if style_option == "non-pitch change":
                    # gTTS 직접 사용 (피치 변화 없음)
                    tts = gTTS(text=korean_text, lang='ko')
                    tts.save(output_path)
                               
                elif style_option == "depth version":
                    # MathSpeechSynthesizer 기본 정책 사용 (피치 변조 적용)
                    synthesizer = MathSpeechSynthesizer()
                    synthesizer.save(expr, output_path=output_path)
                
                elif style_option == "grouping version":
                    latex_audio_grouping_pitch(expr, output_path)
                
                # 재생 및 다운로드 UI
                st.success("생성 완료!")
                st.audio(output_path, format='audio/mp3')
                
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="⬇️ MP3 다운로드",
                        data=file,
                        file_name="math_speech.mp3",
                        mime="audio/mp3"
                    )
                
                # 임시 파일 정리 (선택 사항)
                # os.remove(output_path) 

            except Exception as e:
                st.error(f"음성 합성 중 오류 발생: {e}")
else:
    st.info("수식을 입력하면 미리보기와 변환 결과가 나타납니다.")