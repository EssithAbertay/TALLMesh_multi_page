# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 15:28:10 2024

@author: Laptop
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from api_key_management import manage_api_keys, load_api_keys

def main():
    st.header(":orange[Some explanations]")

    st.subheader("Guidelines:")
    st.write("This software follows the method we have detailed in our papers. It roughly follows the 6 phases approach to Thematic Analysis by Braun and Clarke, but the process is adapated to LLMs and their limits")
    st.divider()
    st.write(":green[** Phase 1 - Familiarise with the data**]")
    st.write("""
       LLMs do not familiarise with the data, but to use the software you will need to prepare your data.
       Let's assume you are analysing intervies. You will need to have the files as **.txt** files. Please also make sure the encoding is UTF-8.
       If you wish, you can cut from the interviews unnecessary parts, such as initial introductions, final salutations etc. Parts which do not contain any relevant data.
       This will help reducing the cost for processing but also avoiding the model generate codes on e.g. final salutations
       
    """)
    st.write("""
       We also recommend naming the interviews with the following format: 1_Interview_[then your keywork e.g. gaming], 2_Interview_gaming.
       This is not essential but it can help you keep track of what you have already analysed.
       
    """)
    st.divider()
    st.write(":green[** Phase 2 - Initial Coding**]")
    st.write("""
       Each interview will be coded independently, one-by-one.
    """)
    st.write("""
       Each set of code (for each interview) will be saved in a **.csv** file.
    """)
    st.write("""
       We offer a pre-populated prompt, which we have tested and should work in most cases, but you are free to change
       the prompt, or adapt the pre-populated prompt.   
    """)
    st.write("""
       There is no guarante that the pre-populated prompt will produce the expected output, due to a variety of reasons.
    """)
    st.write(":blue[Reduction of Initial Coding]")
    st.write("""
       Because each interview is coded independently we recommend to reduce the codebook by identifying duplicates. Our process follows the procedure we have detailed in one of our papers
       and also has some vague resemblance with the CoMeTs method proposed by Costantinou et al.
    """)
    st.write("""
       This will help reduce the number of codes and facilitate the finding of Themes, and it also will allow
       to measreu the Initial Thematic Saturation (ITS)
    """)

    

if __name__ == "__main__":
    main()
