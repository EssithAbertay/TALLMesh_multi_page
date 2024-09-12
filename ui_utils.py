import streamlit as st

# reused UI functions / components

# Function create circled numbers... seems overkill, there must be a simpler way
def create_circle_number(number):
    return f"""
        <div style="
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background-color: orange;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0 auto 20px auto;
        ">
            <span style="
                color: white;
                font-size: 24px;
                font-weight: bold;
            ">{number}</span>
        </div>
    """

# Function to create centered content in a column with a numbered circle
def centered_column_with_number(column, number, title, image):
    with column:
        st.markdown(create_circle_number(number), unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: orange;'>{title}</h3>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)