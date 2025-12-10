import streamlit as st
import os
import tempfile
from gtts import gTTS
import math
import csv            # [추가] CSV 기록용
import shutil         # [추가] 파일 복사용
from datetime import datetime # [추가] 타임스탬프용

from expr_audio_pitch import (
    latex_audio_depth_change,
    latex_audio_grouping_pitch,
    AudioPolicy,
    create_custom_policy
)

from IPython.display import Audio, display
import os

# 가이드라인에 명시된 핵심 모듈 임포트
from LaTeX_Parser import latex_to_expression, test_cases
from Expression_Syntax import expression_to_korean, expression_to_tokens_with_pitch
from speech_synthesizer import MathSpeechSynthesizer

# ----------------- [추가된 함수] 로컬 저장 로직 -----------------
def save_log_local(latex_text, style_mode, src_audio_path):
    """
    생성된 오디오 파일과 메타데이터(수식, 모드, 시간)를 로컬 폴더에 저장합니다.
    """
    # 1. 저장할 기본 디렉토리 설정
    base_dir = "saved_data"
    audio_dir = os.path.join(base_dir, "audio")
    
    # 폴더가 없으면 생성
    os.makedirs(audio_dir, exist_ok=True)
    
    # 2. 파일명 생성 (날짜_시간_스타일.mp3)
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    # 파일명에 공백이 있으면 관리가 어려우므로 _로 대체
    safe_style = style_mode.replace(" ", "_") 
    filename = f"{timestamp_str}_{safe_style}.mp3"
    dest_audio_path = os.path.join(audio_dir, filename)
    
    # 3. 임시 오디오 파일을 영구 저장소로 복사
    try:
        shutil.copy(src_audio_path, dest_audio_path)
    except Exception as e:
        print(f"파일 복사 실패: {e}")
        return

    # 4. CSV 파일에 로그 기록 (saved_data/history_log.csv)
    log_file_path = os.path.join(base_dir, "history_log.csv")
    file_exists = os.path.isfile(log_file_path)
    
    try:
        # utf-8-sig는 엑셀에서 한글 깨짐을 방지합니다.
        with open(log_file_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # 파일이 처음 생성될 때만 헤더 작성
            if not file_exists:
                writer.writerow(["Timestamp", "Style_Mode", "Audio_Filename", "LaTeX_Input"])
            
            # 데이터 한 줄 추가
            writer.writerow([
                now.strftime("%Y-%m-%d %H:%M:%S"), 
                style_mode, 
                filename, 
                latex_text
            ])
            print(f"로그 저장 완료: {filename}")
    except Exception as e:
        print(f"CSV 기록 실패: {e}")

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
    ("standard", "non-pitch change", "depth version", "grouping version"),
    index=0, # 기본값: Expressive
    help="standard: 기본TTS\nnon-pitch changle: 높낮이 없음\nepth version: d 자연스러운 피치\nHierarchical: 구조 강조형"
)

# 2. 구어체 모드 선택
is_naive = st.sidebar.checkbox(
    "구어체 모드 (Casual)",
    value=True,
    help="체크 시: '이 분의 일' (자연스러움)\n해제 시: 형식적인 수학 표현"
)

# 3. 음성 성별 선택
is_male = st.sidebar.selectbox(
    "음성 성별",
    ("male", "female"),
    index=0
)

if is_male == "male":
    is_male = True
else:
    is_male = False

st.sidebar.markdown("### 🔊 오디오 스타일 설정")
pitch_scale = st.sidebar.slider(
    "피치 변화 강도 (Pitch Scale)",
    min_value=0.0,   # 최소 변화량 (0이면 변화 없음)
    max_value=10.0,  # 최대 변화량 (필요에 따라 조절)
    value=2.0,       # 기본값 (기존에 사용하던 수치)
    step=0.5,        # 조절 단위
    help="수식의 깊이(depth)에 따른 음 높낮이 변화 폭을 조절합니다. 값이 클수록 변화가 급격해집니다."
)


st.sidebar.markdown("---")
st.sidebar.info("Dolphin-doing-Math Project\nLatex to Korean Speech")

with st.expander("ℹ️ 튜토리얼: 수식의 구조를 소리로 듣는 법 (여기를 클릭하세요)"):
    st.markdown("### 🎵 피치(Pitch) 변화 원리")
    st.write("""
    이 프로그램은 눈으로 보는 수식의 구조를 귀로 파악할 수 있도록, 
    **수식의 깊이(Depth)**에 따라 목소리의 높낮이를 실시간으로 조절합니다.
    """)
    
    st.divider() # 구분선

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**예시:**")
        st.latex(r"x + \frac{1}{y^2}")
    
    with col2:
        st.markdown("**작동 방식:**")
        st.markdown("""
        1. **기본 톤:** $x +$ (가장 바깥쪽)
        2. **1단계 변화:** 분수 안으로 진입 시 ($1$, $y$) 피치가 변함
        3. **2단계 변화:** 지수 안으로 진입 시 ($^2$) 피치가 더 크게 변함
        """)

    st.info("""
    💡 **팁:** 사이드바의 **'피치 변화 강도'** 슬라이더를 조절하여, 
    깊이에 따른 목소리 변화폭을 나에게 맞게 설정할 수 있습니다.
    """)

with st.expander("📖 Grouping pitch 튜토리얼 (상세 매뉴얼)"):
    st.write("수학 기호는 크게 **원자 값, 전위 연산자, 중위 연산자, 후위 연산자, 서술자**로 구분합니다.")
    
    st.markdown("---") # 구분선

    st.markdown("#### 1) 원자 값 (Atomic value)")
    st.write("더 이상 분해하지 않고 그 자체로 항(operand)이 되는 최소 단위입니다.")
    st.markdown("- **예시:** 숫자 $3$, 변수 $x$, 상수 $e$, $\pi$, $\emptyset$, 무한($\infty$) 등")

    st.markdown("#### 2) 전위 연산자 (Prefix / unary operator)")
    st.write("피연산자보다 연산자를 먼저 읽는 연산자입니다.")
    st.markdown("- **예시:** $|x|$ (절댓값), $[x]$ (가우스), $\sqrt{x}$ (루트), $\sin$ (사인), $+x$, $-x$ 등")

    st.markdown("#### 3) 중위 연산자 (Infix / binary operator)")
    st.write("두 항 사이에 위치해서 두 항을 결합하는 연산자로, 읽을 때에도 두 피연산자 중간에 읽습니다.")
    st.markdown("""
    **예시:**
    - $a+b$
    - $A \cap B$
    - $a:b:c$
    """)

    st.markdown("#### 4) 후위 연산자 (Postfix operator)")
    st.write("피연산자 뒤에 붙어서 피연산자를 먼저 말하고, 연산자를 말합니다.")
    st.markdown("""
    **예시:**
    - $n!$
    - $f'$
    - $x_1$, $x^2$
    """)

    st.markdown("#### 5) 서술자 (Descriptive / relational operator)")
    st.write("값을 만들어내는 연산이라기보다, **문장(명제)**을 만듭니다.")
    st.markdown("""
    - **비교/관계:** $=, \\neq, <, \leq, >$
    - **집합 관계:** $\in, \\notin, \subset, \subseteq, \supseteq$
    - **논리/함의 관계:** $\\to, \Rightarrow, \iff, \Leftrightarrow$
    - **기하 관계:** $\parallel, \perp, \equiv, \sim$
    """)
    st.markdown("""
    **예시:**
    - $a=b$
    - $x \in A$
    - $l \perp m$
    """)

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
            
            # 최종적으로 재생할 파일의 경로를 담을 변수
            final_audio_path = None

            try:
                # 1. gTTS 계열 (표준, 높낮이 없음) - 임시 파일 필요
                if style_option in ["non-pitch change", "standard"]:
                    # gTTS는 경로를 리턴하지 않고 직접 저장하므로 임시 파일을 미리 만듭니다.
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                        temp_path = tmp_file.name
                    
                    if style_option == "non-pitch change":
                        tts = gTTS(text=korean_text, lang='ko')
                        tts.save(temp_path)
                    elif style_option == "standard":
                        tts = gTTS(text=latex_input, lang='ko')
                        tts.save(temp_path)
                    
                    # 저장된 임시 파일 경로를 최종 경로로 설정
                    final_audio_path = temp_path

                # 2. 커스텀 계열 (Depth, Grouping) - 함수가 경로를 리턴함
                elif style_option == "depth version":
                    # [중요] 함수가 반환하는 '진짜 경로'를 받습니다. 임시 파일 경로는 넘기지 않아도 됩니다(라이브러리에서 알아서 함).
                    final_audio_path = latex_audio_depth_change(
                        latex_input, 
                        is_male=is_male, 
                        is_naive=is_naive,
                        filename="depth_ver.mp3" # 식별용 이름 (UUID 자동 부착됨)
                    )
                
                elif style_option == "grouping version":
                    # [중요] 리턴값을 받아야 재생 가능!
                    final_audio_path = latex_audio_grouping_pitch(
                        latex_input, 
                        is_male=is_male, 
                        is_naive=is_naive,
                        filename="grouping_ver.mp3"
                    )

                # ---------------------------------------------------------
                # 공통 재생 및 저장 로직
                # ---------------------------------------------------------
                
                if final_audio_path and os.path.exists(final_audio_path):
                    st.success("생성 완료!")
                    
                    # 1. 로컬에 백업 저장 (로그 기록)
                    save_log_local(latex_input, style_option, final_audio_path)

                    # 2. 파일을 바이너리로 읽어서 재생 (브라우저 권한 문제 해결)
                    with open(final_audio_path, "rb") as f:
                        audio_bytes = f.read()
    
                    # 확장자 확인
                    file_ext = os.path.splitext(final_audio_path)[1].lower()
                    mime_type = "audio/wav" if "wav" in file_ext else "audio/mp3"
                    
                    # 플레이어 표시
                    st.audio(audio_bytes, format=mime_type)
                    
                    # 다운로드 버튼
                    st.download_button(
                        label="⬇️ MP3 다운로드",
                        data=audio_bytes,
                        file_name=os.path.basename(final_audio_path),
                        mime=mime_type
                    )
                    
                    # (선택) gTTS로 만든 임시 파일인 경우에만 삭제 (캐시 파일은 유지)
                    # if style_option in ["non-pitch change", "standard"]:
                    #    os.remove(final_audio_path)

                else:
                    st.error("오디오 파일이 생성되지 않았거나 경로를 찾을 수 없습니다.")
                    # 디버깅용: 경로가 뭐로 잡혔는지 확인
                    st.write(f"Debug: final_audio_path = {final_audio_path}")

            except Exception as e:
                st.error(f"음성 합성 중 오류 발생: {e}")
                import traceback
                st.text(traceback.format_exc()) # 상세 에러 로그 출력