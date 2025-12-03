import streamlit as st
import os
import tempfile
from gtts import gTTS
import math

# 가이드라인에 명시된 핵심 모듈 임포트
from LaTeX_Parser import latex_to_expression, test_cases
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

# [Session State 초기화]
# 입력창의 값을 저장하고 버튼과 동기화하기 위한 변수입니다.
if "target_latex" not in st.session_state:
    # 초기값 설정 (리스트가 비어있지 않다면 첫 번째 케이스 사용)
    if test_cases and isinstance(test_cases[0], (tuple, list)):
        st.session_state["target_latex"] = test_cases[0][0]
    else:
        st.session_state["target_latex"] = r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}"

# ----------------- [UI 1] Test Cases 선택 UI -----------------
with st.expander("📚 테스트 케이스 (Test Cases) 선택 패널", expanded=True):
    st.caption("아래 번호를 클릭하면 해당 수식이 자동으로 입력됩니다.")
    
    # 1. 페이지 계산 (15개씩 분할)
    BATCH_SIZE = 15
    total_items = len(test_cases)
    total_pages = math.ceil(total_items / BATCH_SIZE)

    # 2. 범주(페이지) 선택 박스
    # 예: "Section 1 (1~15)", "Section 2 (16~30)" ...
    page_options = [f"Section {i+1} ({i*BATCH_SIZE + 1} ~ {min((i+1)*BATCH_SIZE, total_items)})" for i in range(total_pages)]
    selected_page = st.selectbox("범주 선택", page_options, label_visibility="collapsed")

    # 3. 버튼 생성 및 이벤트 처리
    if selected_page:
        page_idx = page_options.index(selected_page)
        start_idx = page_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_items)
        
        # 현재 페이지에 해당하는 데이터 슬라이싱
        current_batch = test_cases[start_idx:end_idx]
        
        # 5열 그리드로 버튼 배치
        cols = st.columns(5)
        
        for i, item in enumerate(current_batch):
            real_idx = start_idx + i + 1
            
            # [핵심 변경 사항] item이 (latex, ast) 튜플이므로 첫 번째 요소 추출
            if isinstance(item, (tuple, list)):
                latex_code = item[0]
            else:
                latex_code = str(item) # 만약 튜플이 아닌 문자열만 있는 경우 대비

            with cols[i % 5]:
                # 버튼 라벨: "No. 1", "No. 2" ...
                if st.button(f"No. {real_idx}", key=f"btn_{real_idx}", use_container_width=True):
                    st.session_state["target_latex"] = latex_code
                    st.rerun()

st.markdown("---")
# 입력창
latex_input = st.text_area(
    "LaTeX 수식을 입력하세요:",
    value=r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}",
    height=120,
    key="target_latex"
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