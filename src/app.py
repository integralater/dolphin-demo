# app.py 파일 내용

import streamlit as st
import os
import tempfile
import time # 로딩 표시를 위해 추가
# 1단계에서 분리한 핵심 로직을 가져옵니다.
from Expression_Syntax import *
from LaTeX_Paser import *
from gtts_expr_audio_pitch import *

# ----------------- A. 페이지 설정 -----------------
st.set_page_config(
    page_title="LaTeX 음성 변환 데모", # 브라우저 탭에 표시되는 제목
    layout="wide"
)

# ----------------- B. 제목 및 설명 (정적인 부분) -----------------
st.title("🔢 LaTeX 수식 음성 변환 데모")
st.markdown("수식 **구조적 깊이**에 따라 피치(음높이)가 변조된 한국어 음성 파일을 생성합니다.")
st.markdown("---")


# ----------------- C. 입력 위젯 만들기 -----------------

# 사용자가 LaTeX 코드를 입력할 수 있는 큰 텍스트 상자를 만듭니다.
latex_input = st.text_area(
    "여기에 LaTeX 수식을 입력하세요:",
    value=r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}", # 기본 예시 수식
    height=150
)

st.subheader("일반 수식 표기 (실시간 미리보기)")

if latex_input.strip():
    # 📌 실시간 변환 로직 (버튼 클릭과 무관하게 실행됨)
    try:
        parser = LatexParser(latex_input)
        # AST 생성
        ast_root = parser.parse_full()
        # AST를 사람이 읽기 쉬운 문자열로 변환 (Expression.__str__ 사용)
        human_readable_latex = str(ast_root)
        
        # Streamlit의 st.latex는 LaTeX 코드를 렌더링하여 보여줍니다.
        # 
        st.latex(latex_input)
        
        # 파싱 결과를 일반 텍스트로도 보여줄 수 있습니다.
        st.caption(f"파싱된 내부 구조 (Repr): {repr(ast_root)}")

    except Exception as e:
        # 파싱 오류 시에는 오류 메시지 출력
        st.error(f"❌ 수식 파싱 오류: {e}")
else:
    st.info("수식을 입력하면 여기에 일반 수식 미리보기가 나타납니다.")

st.markdown("---")
st.subheader("음성 변환 및 재생")
# 변환을 시작하는 버튼을 만듭니다.
if st.button("🔊 음성 변환 및 재생 시작"):
    
    if not latex_input.strip():
        st.error("LaTeX 수식을 입력해주세요!")
    else:
        # st.spinner를 사용하면 변환 중이라는 로딩 애니메이션이 표시됩니다.
        with st.spinner('변환 중... (gTTS 음성 합성 및 오디오 변조 작업 진행)'):
            
            # ----------------- D. 핵심 로직 실행 (4단계 알고리즘) -----------------
            
            # 임시 디렉토리를 만들어 오디오 파일을 저장합니다.
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_filename = "output_audio.mp3"
                temp_filepath = os.path.join(tmpdir, temp_filename)
                
                try:
                    # 핵심 함수 호출 (이 함수가 10단계 알고리즘을 실행합니다)
                    final_output_path = latex_audio_grouping_pitch(
                        latex_str=latex_input,
                        output_dir=tmpdir,
                        filename=temp_filename
                    )
                    
                    # ----------------- E. 결과 표시 (오디오 재생) -----------------
                    
                    st.success("✅ 음성 변환 완료! 아래에서 재생하세요.")
                    
                    # Streamlit 내장 오디오 플레이어 위젯
                    st.audio(final_output_path, format='audio/mp3')
                    
                    # 다운로드 버튼
                    with open(final_output_path, "rb") as file:
                        st.download_button(
                            label="⬇️ MP3 파일 다운로드",
                            data=file,
                            file_name="math_audio.mp3",
                            mime="audio/mp3"
                        )

                except Exception as e:
                    # 에러가 발생했을 때 사용자에게 알립니다.
                    st.error(f"❌ 변환 중 오류가 발생했습니다. 수식 형식을 확인해 주세요: {e}")
                    # 개발자를 위해 상세 에러 내용도 출력
                    st.exception(e)