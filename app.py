import streamlit as st
import streamlit.components.v1 as components
from predict import ReviewClf
import tempfile

COMPONENT_HEIGHT = 600

st.title('Reviews with important words')

st.markdown('Put review here')

with st.form('form'):
    review_input = st.text_area('Enter review text here: ', 'This is my review. I really like this place')
    submit_button = st.form_submit_button('Click on me')

    if submit_button:
        model = ReviewClf()
        review, fig, exp = model.predict_review(review_input)
        st.write(f'Important words with review:')
        if fig:
            components.html(review)
            st.pyplot(fig)

        st.text(exp.as_list())
        with tempfile.TemporaryDirectory() as temp_dir:
            exp.save_to_file(f'{temp_dir}/lime.html')
            with open(f'{temp_dir}/lime.html', 'r') as f:
                html_content = f.read()
        components.html(html_content, height=COMPONENT_HEIGHT)
