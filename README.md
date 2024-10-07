# TALLMesh: Thematic Analysis with Large Language Models

TALLMesh is an experimental tool that leverages AI assistance to conduct qualitative analysis on text-based data, specifically emulating stages from the Braun & Clarke approach to inductive thematic analysis.

It uses streamlit for the GUI elements, and for the analysis it uses code and ideas from these two papers:

1. [De Paoli, S. (2023). *Performing an Inductive Thematic Analysis of Semi-Structured Interviews With a Large Language Model: An Exploration and Provocation on the Limits of the Approach*. Social Science Computer Review, 08944393231220483.](https://journals.sagepub.com/doi/full/10.1177/08944393231220483)

2. [De Paoli, S., & Mathis, W. S. (2024). *Reflections on Inductive Thematic Saturation as a potential metric for measuring the validity of an inductive Thematic Analysis with LLMs*. arXiv preprint arXiv:2401.03239.](https://arxiv.org/ftp/arxiv/papers/2401/2401.03239.pdf)


## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Configuration](#configuration)
7. [Contributing](#contributing)
8. [License](#license)
9. [Acknowledgements](#acknowledgements)

## Introduction

TALLMesh is designed to assist researchers, academics, and qualitative data analysts in performing thematic analysis using large language models. The tool follows the six-phase approach to Thematic Analysis by Braun & Clarke, adapted for use with LLMs and their capabilities.

This project is based on research conducted by Prof. Stefano De Paoli and Dr. Daniel Rough, with initial support from the British Academy for designing the Graphical User Interface.

## Features

- Project-based organization for managing multiple analyses
- Support for multiple LLM providers (OpenAI, Anthropic, Azure)
- Streamlined workflow emulating phases 2, 3, and 4 of Braun & Clarke's six-phase approach:
  1. Familiarization with data
  2. Initial coding
  3. Searching for themes
  4. Reviewing themes
  5. Defining and naming themes
  6. Producing the report
- Interactive visualizations including icicle charts and thematic network maps
- Calculation of Inductive Thematic Saturation (ITS) metric
- Flexible prompt engineering for each analysis phase
- Export options for codes, themes, and visualizations

## TALLMesh Process

- Project Set Up

Use this page to set up your project and add. [video to come]

- Initial Coding

This page emulates Braun & Clarke's phase 2, in which initial codes are derived from each uploaded .txt file

- Reduction of Codes

This stage involves the reduction of duplicate codes to yield a list of unique codes

- Finding Themes

During this stage the LLM is tasked with identifying patterns and grouping associated codes together under broader themes.


## Installation

- Non-technical users

We've created custom GPTs to help you with different stages of the thematic analysis process:

1. [TALLMesh 1: Initial Codes](https://chatgpt.com/g/g-zgTm3PdhK-thematic-analysis-tallmesh-1-initial-codes) - Assists with the initial coding phase.
2. [TALLMesh 2: Code Reduction](https://chatgpt.com/g/g-tn0CEC5Am-thematic-analysis-tallmesh-2-code-reduction) - Helps with reducing and consolidating codes.
3. [TALLMesh 3: Find Themes](https://chatgpt.com/g/g-IP7o7e2SD-thematic-analysis-tallmesh-3-find-themes) - Guides you through the theme identification process.

These custom GPTs provide a user-friendly interface to perform thematic analysis without needing to set up the full TALLMesh environment.

- For those with some Python familiarity

1. Clone the repository:

   ```
   git clone https://github.com/flipflop4/TALLMesh_multi_page.git 
   cd tallmesh_multi_page
   ```

2. Create a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit app:

   ```
   streamlit run Thematic_Analysis_LLMs.py
   ```

2. Follow the on-screen instructions to:
   - Set up a new project
   - Upload and manage your data files
   - Configure API keys for LLM providers
   - Perform each phase of the thematic analysis
   - Generate visualizations and reports

## Project Structure

- `Thematic_Analysis_LLMs.py`: Main entry point for the Streamlit application
- `1_🏠_Project_Set_Up.py`: Project and file management
- `2_1️⃣_Initial_coding.py`: Initial coding phase
- `3_2️⃣_Reduction_of_codes.py`: Code reduction and consolidation
- `4_3️⃣_Finding_Themes.py`: Theme identification
- `5_💹_Saturation_Metric.py`: Calculation of saturation metrics
- `6_🔗_Thematic_Overlap_Map.py`: Thematic network visualization
- `7_🧊_Theme-Codes_Icicle.py`: Icicle chart visualization
- `8_🕸️_Spider_Diagram.py`: Spider diagram visualisation
- `9_🌳_Nested_Treemap.py`: Nested treemap visualisation
- `10_💡_Resources.py`: Overview of Thematic Analysis & Comparison
- `11_⚙️_Azure_Settings.py`: Manage Azure API credentials
- `12_📌_Guide.py`: User guide
- `13_📢_Prompt Settings.py`: Prompt configuration settings page
- `14_📤_File_Upload_and_Conversion.py`: Convert docs (PDF, DOCX) to.txt 

## Configuration

1. API Keys: Use the sidebar in the application to manage API keys for different LLM providers.
2. Azure Settings: If using Azure, configure the deployment settings in the application (there is a dedicated page for Azure settings)

## Contributing

We welcome contributions to TALLMesh! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with clear, descriptive messages
4. Push your changes to your fork
5. Submit a pull request with a clear description of your changes

## License

TBC

## Acknowledgements

- Prof. Stefano De Paoli and Dr. Daniel Rough for their research and development of this tool
- The British Academy for their support 
- Virginia Braun and Victoria Clarke for their seminal work on Thematic Analysis

For questions or suggestions, please contact the project PIs: Prof. Stefano De Paoli (s.depaoli@abertay.ac.uk) & Dr. Daniel Rough (drough001@dundee.ac.uk).

---
