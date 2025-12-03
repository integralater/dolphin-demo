import streamlit as st

if 'message' not in st.session_state:
    st.session_state.message = ''
if 'selected_range' not in st.session_state:
    st.session_state.selected_range = None

st.title('메시지 템플릿 선택')

# 버튼 텍스트 크기를 절반으로 줄이기 위한 CSS
st.markdown("""
<style>
    button[kind="primary"] {
        font-size: 0.125em !important;
    }
    .stButton > button {
        font-size: 0.25em !important;
        padding: 0.05rem 0.3rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
</style>
""", unsafe_allow_html=True)

# 텍스트 영역
message = st.text_area(
    '메시지 작성:', 
    value=st.session_state.message,
    height=150
)

# test_cases 섹션
st.markdown('### test_cases')

# 범주 선택 (15개씩 나눔)
total_buttons = 169
range_size = 15
ranges = []
for start in range(1, total_buttons + 1, range_size):
    end = min(start + range_size - 1, total_buttons)
    ranges.append(f'{start}~{end}')

# selectbox의 index 계산
if st.session_state.selected_range is None:
    default_index = 0
elif st.session_state.selected_range in ranges:
    default_index = ranges.index(st.session_state.selected_range) + 1
else:
    default_index = 0

selected_range = st.selectbox(
    '범주 선택:',
    options=['-'] + ranges,
    index=default_index,
    key='range_selector'
)

# 선택된 범주 저장
if selected_range != '-':
    st.session_state.selected_range = selected_range
else:
    st.session_state.selected_range = None

# 컬럼으로 버튼 배치 (169개 버튼, 1부터 169까지)
# 선택된 범주에 따라 버튼 표시
if st.session_state.selected_range is not None:
    # 선택된 범주 파싱
    start_range, end_range = map(int, st.session_state.selected_range.split('~'))
    
    num_cols = 15
    cols = st.columns(num_cols)
    
    button_count = 0
    for i in range(start_range - 1, end_range):
        col_idx = button_count % num_cols
        button_num = i + 1
        with cols[col_idx]:
            if st.button(str(button_num), key=f'btn_{button_num}'):
                st.session_state.message = str(button_num)
        button_count += 1

#latex 렌더링 예제
# 최종 메시지 표시
if message:
    st.success(f'작성된 메시지: {message}')

st.title('LaTeX 수식 렌더링 예제')

# 탭으로 구분
tab1, tab2, tab3 = st.tabs(['기본 수식', '복잡한 수식', '행렬'])

with tab1:
    st.subheader('기본 수식들')
    st.latex(r'a^2 + b^2 = c^2')
    st.latex(r'\sum_{i=1}^{n} i = \frac{n(n+1)}{2}')
    
with tab2:
    st.subheader('적분과 미분')
    st.latex(r'\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}')
    st.latex(r'\frac{d}{dx}(x^n) = nx^{n-1}')
    
with tab3:
    st.subheader('행렬')
    st.latex(r'''
        A = \begin{bmatrix}
        a_{11} & a_{12} & a_{13} \\
        a_{21} & a_{22} & a_{23} \\
        a_{31} & a_{32} & a_{33}
        \end{bmatrix}
    ''')