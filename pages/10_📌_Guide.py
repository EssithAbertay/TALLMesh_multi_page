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
High level overview:
:orange[1.] Set up your project and upload your data
:orange[2.] Conduct initial coding phase
:orange[3.] Reduction of duplicate/highly similar codes
:orange[4.] Generate themes from reduced codes
:orange[5.] Finalise theme book
:orange[6.] Metrics and visualisations

Each page of the application corresponds to one of the stages outlined above. 
Let's explore each of these in more detail...
""")

st.divider()

st.header(":orange[Project and File Management]")

st.write("""
The "Folder Set Up" page is where you'll begin your thematic analysis journey. This page allows you to create new projects, manage existing ones, and upload the files you want to analyze. Here's how to use it:
""")

st.subheader(":orange[1. Creating a New Project]")
st.write("""
- Enter a unique name for your project in the "Enter new project name:" text box.
- Click the "Create Project" button to set up your project.
- The system will create a new folder structure for your project, including subfolders for data, initial codes, reduced codes, themes, and more.
""")

st.subheader(":orange[2. Selecting an Existing Project]")
st.write("""
- Use the dropdown menu labeled "Select a project:" to choose from your existing projects.
- Once selected, you'll see the project name displayed and have options to manage its files.
""")

st.subheader(":orange[3. Uploading Files]")
st.write("""
- With a project selected, you'll see a file uploader labeled "Upload interviews .txt files".
- You can drag and drop multiple .txt files or click to browse and select them.
- Click the "Upload Files" button to add these files to your project's data folder.
- Successful uploads will be confirmed with a message.
""")

st.subheader(":orange[4. Managing Existing Files]")
st.write("""
- Below the file uploader, you'll see a list of all files currently in your project.
- Each file has a checkbox next to it. Select the checkbox for any files you want to delete.
- Click the trash can icon (🗑️) that appears to remove selected files from your project.
""")

st.subheader(":orange[5. Deleting a Project]")
st.write("""
- If you need to remove an entire project, select it from the dropdown and click the "Delete Project" button.
- This action will remove all files and folders associated with the project, so use it carefully.
""")

st.subheader(":orange[6. API Key Management]")
st.write("""
- In the sidebar, you'll find options to manage your API keys for different AI providers.
- You can add new keys, view existing ones (last few digits only for security), and delete keys as needed.
""")


st.subheader(":orange[Tips]")
st.write("""
- Always double-check your project selection before uploading or deleting files.
- It's recommended to use descriptive names for your projects to easily identify them later.
- If you're analyzing interviews, consider naming your files consistently (e.g., "1_Interview_topic.txt", "2_Interview_topic.txt").
- Remember that deleting files or projects is permanent and cannot be undone.
""")

st.info(":bulb: By properly setting up your project and managing your files here, you'll have a solid foundation for the subsequent steps in your thematic analysis process.")

st.divider()

st.header(":orange[Initial Coding]")

st.write("""
The Initial Coding page is where you begin the analysis of your data. This step involves generating initial codes for each of your uploaded files using AI assistance. Here's how to use this page:
""")

st.subheader(":orange[1. Project Selection]")
st.write("""
- Use the dropdown menu to select the project you want to work on.
- If you haven't set up a project yet, you'll be prompted to go to the 'Folder Set Up' page first.
""")

st.subheader(":orange[2. File Selection]")
st.write("""
- Once a project is selected, you'll see a list of files available for processing.
- You can select individual files or use the "Select All" checkbox to choose all files at once.
- :orange[Files that have already been processed will be marked with a warning icon.]
""")

st.subheader(":orange[3. LLM Settings]")
st.write("""
- Choose the AI model you want to use for the analysis from the dropdown menu. 
- Select a preset prompt or edit the provided prompt to customize your analysis.
- Adjust the model temperature and top_p values using the sliders. These parameters control the creativity and randomness of the AI's output.
""")

st.info("Make sure you have provided an API key for the provider of the model you have selected (e.g., Anthropic, Azure, OpenAI)")

st.subheader(":orange[4. Processing Files]")
st.write("""
- After configuring your settings, click the "Process" button to start the initial coding.
- :orange[The system will process each selected file and generate initial codes.]
- A progress bar will show you the status of the processing.
""")

st.subheader(":orange[5. Viewing Results]")
st.write("""
- Once processing is complete, you'll see the results for each file in expandable sections.
- Each section will show a table with the generated codes, their descriptions, and relevant quotes.
- You can download the results for each file individually using the provided download buttons.
""")

st.subheader(":orange[6. Saved Initial Codes]")
st.write("""
- At the bottom of the page, you'll find an expandable section showing previously processed files.
- You can view, delete, or download these saved initial codes.
""")

st.subheader(":orange[Tips]")
st.write("""
- :orange[Experiment with different prompts and model settings to get the best results for your data.]
- If you're not satisfied with the initial codes, you can delete them and reprocess the file with different settings.
- Remember that this is just the first step in the analysis process. The codes generated here will be refined in later stages.
""")

st.info("Initial coding sets the foundation for your thematic analysis. Take your time to review the generated codes and ensure they capture the essence of your data before moving on to the next stage.")

st.divider()

st.header(":orange[Reduction of Codes]")

st.write("""
The Reduction of Codes page is where you refine and consolidate the initial codes generated in the previous step. This process helps to identify patterns and reduce redundancy in your coding. Here's how to use this page:
""")

st.subheader(":orange[1. Project and File Selection]")
st.write("""
- Select your project from the dropdown menu.
- Once a project is selected, you'll see a list of files containing initial codes.
- Choose the files you want to process. You can select individual files or use the "Select All" checkbox.
""")

st.subheader(":orange[2. LLM Settings]")
st.write("""
- Choose the AI model you want to use for code reduction.
- Select a preset prompt or edit the provided prompt to guide the reduction process.
- Adjust the model temperature and top_p values using the sliders. These parameters influence the AI's output.
""")

st.subheader(":orange[3. Processing and Results]")
st.write("""
- Click the "Process" button to start the code reduction.
- The system will analyze the selected files sequentially, comparing and merging similar codes.
- A progress bar will show the status of the processing.
- Once complete, you'll see:
  - A table of reduced codes with their descriptions, merged explanations, and associated quotes.
  - A "Code Reduction Results" table showing the number of total and unique codes for each processed file.
- You can download both the reduced codes and the code reduction results as CSV files.
""")

st.subheader(":orange[4. Saved Reduced Codes]")
st.write("""
- At the bottom of the page, you'll find an expandable section showing previously processed reduced code files.
- You can view, delete, or download these saved reduced codes.
""")

st.subheader(":orange[Key Features]")
st.write("""
- :orange[Automatic merging:] The AI identifies similar codes and combines them, providing explanations for the merges.
- :orange[Quote preservation:] All quotes associated with the original codes are retained and linked to the reduced codes.
- :orange[Tracking changes:] The system keeps track of how initial codes map to reduced codes, maintaining traceability.
- :orange[Saturation analysis:] The code reduction results can be used to assess thematic saturation in your analysis (see 'Saturation Metric).
""")

st.subheader(":orange[Tips]")
st.write("""
- Review the merged codes carefully to ensure the AI's decisions align with your understanding of the data.
- If you're not satisfied with the reduction, you can adjust the prompt or model settings and reprocess the files.
- :orange[Pay attention to the "Code Reduction Results" table.] A plateauing number of unique codes may indicate approaching saturation in your analysis.
- Consider running the reduction process multiple times with different settings to compare results and ensure thorough analysis.
""")

st.info("Code reduction is a critical step in refining your analysis. It helps to consolidate your findings and prepare for the identification of overarching themes in the next stage.")

st.divider()

st.header(":orange[Finding Themes]")

st.write("""
The Finding Themes page is where you identify overarching themes from your reduced codes. This step helps you synthesize your data into meaningful patterns. Here's how to use this page:
""")

st.subheader(":orange[1. Project and File Selection]")
st.write("""
- Select your project from the dropdown menu.
- Once a project is selected, you'll see a list of reduced code files available for processing.
- Choose the files you want to analyze. You can select individual files or use the "Select All" checkbox.
""")

st.subheader(":orange[2. LLM Settings]")
st.write("""
- Choose the AI model you want to use for theme identification.
- Select a preset prompt or edit the provided prompt to guide the theme finding process.
- Adjust the model temperature and top_p values using the sliders. These parameters influence the AI's creativity and output variability.
""")

st.subheader(":orange[3. Processing and Results]")
st.write("""
- Click the "Process" button to start finding themes.
- The system will analyze the selected reduced code files and generate themes.
- Once complete, you'll see:
  - An expandable section for each generated theme, showing the theme name, description, and associated codes.
  - A reference section showing all codes and their descriptions used in the analysis.
- You can download the generated themes as a CSV file.
""")

st.subheader(":orange[4. Saved Themes]")
st.write("""
- At the bottom of the page, you'll find an expandable section showing previously generated theme files.
- You can view, delete, or download these saved theme files.
""")

st.subheader("Key Features")
st.write("""
- :orange[Automated theme generation:] The AI identifies patterns across your reduced codes to suggest overarching themes.
- :orange[Theme descriptions:] Each theme comes with a detailed description to explain its meaning and relevance.
- :orange[Code mapping:] The system shows which codes are associated with each theme, maintaining the connection between your data and the higher-level themes.
- :orange[Flexibility:] You can adjust the prompt and model settings to influence how themes are generated and organized.
""")

st.subheader(":orange[Tips]")
st.write("""
- Review the generated themes carefully. While the AI is helpful, your expertise and understanding of the context are crucial for validating and refining these themes.
- :orange[Experiment with different prompts and settings] if you're not satisfied with the initial results. Different approaches can yield different insights.
- Consider the number of themes generated. Too few might oversimplify your data, while too many might make it difficult to draw meaningful conclusions.
- Use the reference section to understand how individual codes contribute to the larger themes.
- Remember that theme generation is an iterative process. You may need to run this step multiple times, adjusting your approach based on the results.
""")

st.info("Finding themes is a crucial step in synthesizing your analysis. It helps you move from detailed codes to broader, more conceptual understanding of your data. Take your time to reflect on the themes and how they relate to your research questions.")