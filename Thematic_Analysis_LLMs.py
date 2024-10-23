# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 08:55:30 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""
import streamlit as st




st.set_page_config(
    page_title="**Thematic Analysis with LLMs**",
    page_icon="📝",
   # page_icon="images/logo.jpg",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        
        'About': "# This is an experimental research project offering a solution for TA with LLM. It is not a software in production"
        }
)

logo = "pages/static/tmeshlogo.png"
st.logo(logo)


st.header(":orange[TALLMesh: Thematic Analysis with Large Language Models]")
#st.sidebar.success("Select a page above.")

st.subheader("Welcome to this 🅰️lpha version")
# subheader
st.write("It allows you to perfom phase 2, phase 3 and phase 4 of Thematic Analysis following Braun and Clarke 6 phases")

st.divider()
st.write("It follows the aproach we detailed in the papers below. **:green[ If you use our software please remember to cite our work :+1:] **")
#st.write("*** If you use our software please remember to cite our work ***")

# display text in italic formatting
st.markdown("📃 De Paoli, S. (2023). Performing an Inductive Thematic Analysis of Semi-Structured Interviews With a Large Language Model: An Exploration and Provocation on the Limits of the Approach. Social Science Computer Review, 08944393231220483. [Read](https://journals.sagepub.com/doi/full/10.1177/08944393231220483) ")
st.markdown("📃 De Paoli, S., & Mathis, W. S. (2024). Reflections on Inductive Thematic Saturation as a potential metric for measuring the validity of an inductive Thematic Analysis with LLMs. arXiv preprint arXiv:2401.03239. [Read](https://arxiv.org/ftp/arxiv/papers/2401/2401.03239.pdf) ")
st.divider()
st.write ("This is a research project and the interface and underlying scripts are only experimental and offered as they are")
st.write ("For questions or suggestions please contact the PIs: Prof. Stefano De Paoli & Dr. Daniel Rough")
st.divider()
st.write("We received initial support from the British Academy to design the Graphical User Interface.")
st.divider()
st.write("The software is available under MIT license [Read]https://opensource.org/license/mit") 


