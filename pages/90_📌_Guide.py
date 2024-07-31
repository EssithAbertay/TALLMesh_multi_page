# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 15:28:10 2024

@author: Laptop
"""

import streamlit as st

st.header(":orange[Welcome to TALLMesh: Thematic Analysis with Large Language Models]")

st.write("""
This application is designed to assist researchers and students in conducting thematic analysis 
using large language models (LLMs). Our approach is inspired by the widely-used Braun and Clarke 
method, adapted for the capabilities of modern AI.

TALLMesh aims to streamline the process of qualitative data analysis, offering a novel way to 
interact with your research data. While it leverages the power of LLMs, it's important to note 
that this tool is meant to augment, not replace, the researcher's critical thinking and interpretive skills.

:blue[Key features of TALLMesh include:]
- Automated initial coding
- Assistance in reducing and refining codes
- Support in identifying and defining themes
- Integration with popular LLM providers

As you use this tool, remember that the AI is a collaborator in your research process. Always 
critically evaluate the outputs and use your expertise to guide the analysis.
""")

st.info("""
:bulb: **Note for researchers:** While TALLMesh can significantly speed up parts of the thematic 
analysis process, it's crucial to maintain rigorous oversight. The tool's suggestions should be 
thoroughly reviewed and adjusted based on your research context and objectives.
""")

st.subheader(":orange[Getting Started]")

st.write("""
To begin using TALLMesh, follow these steps:
1. Set up your project and upload your data
2. Configure your LLM settings
3. Start with initial coding
4. Move through the subsequent phases of analysis

Each section of the application corresponds to a phase in the thematic analysis process. 
Let's explore each of these in more detail...
""")