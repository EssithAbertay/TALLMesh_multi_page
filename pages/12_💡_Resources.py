import streamlit as st
from api_key_management import manage_api_keys, load_api_keys

def main():
    st.header(":orange[Thematic Analysis Resources]")

    st.write("""
    This guide outlines the six phases of Braun and Clarke's approach to thematic analysis 
    and explains how our app aligns with these stages, including some deviations and additional steps.
    """)

    st.subheader(":green[Phase 1 - Familiarizing Yourself with Your Data]")
    
    with st.expander("Braun and Clarke's Approach"):
        st.write("""
        This phase involves immersing yourself in the data, reading and re-reading the material, 
        and noting down initial ideas.
        """)

    with st.expander("Our App's Approach"):
        st.write("""
        - LLMs do not familiarize themselves with the data in the same way humans do.
        - Users need to prepare their data for analysis:
          - Save interview transcripts as .txt files with UTF-8 encoding (see the 📤 File Upload and Conversion page).
          - Optional: Remove unnecessary parts like introductions or final salutations to reduce processing costs and avoid generating irrelevant codes.
                 - Removing interviewer text should be done carefully, or not at all, to avoid removing important context.
          - Recommended naming convention: "1_Interview_[keyword]", "2_Interview_[keyword]", etc.
        """)

    st.divider()

    st.subheader(":green[Phase 2 - Generating Initial Codes]")
    
    with st.expander("Braun and Clarke's Approach"):
        st.write("""
        This phase involves coding interesting features of the data systematically across 
        the entire data set and collating data relevant to each code.
        """)

    with st.expander("Our App's Approach"):
        st.write("""
        - Each interview is coded independently, one-by-one.
        - Initial codes for each interview are saved in a .csv file.
        - We provide pre-populated prompts that users can modify if needed (or create your own from the 📢 Prompt Settings page)
        - Note: The pre-populated prompt may not always produce the expected output due to various factors.
        """)

    with st.expander("Additional Step: Reduction of Initial Coding"):
        st.write("""
        - This step helps reduce the number of codes and facilitates theme identification.
        - It follows the procedure detailed in our papers and has some resemblance to the CoMeTs method.
        - This step also allows for measuring Initial Thematic Saturation (ITS).
        """)

    st.divider()

    st.subheader(":green[Phase 3 - Searching for Themes]")
    
    with st.expander("Braun and Clarke's Approach"):
        st.write("""
        This phase involves collating codes into potential themes and gathering all data 
        relevant to each potential theme.
        """)

    with st.expander("Our App's Approach"):
        st.write("""
        - The app analyzes the reduced codes to identify overarching themes.
        - Users can adjust prompts and model settings to influence theme generation.
        - The system shows which codes are associated with each theme, maintaining the connection between data and higher-level themes.
        """)

    st.divider()

    st.subheader(":green[Phase 4 - Reviewing Themes]")
    
    with st.expander("Braun and Clarke's Approach"):
        st.write("""
        This phase involves checking if the themes work in relation to the coded extracts 
        and the entire data set, generating a thematic 'map' of the analysis.
        """)

    with st.expander("Our App's Approach"):
        st.write("""
        - Users can review the generated themes and their associated codes.
        - The app provides visualizations like the Theme-Codes Icicle and Thematic Network Map to help review the relationships between themes, codes, and data.
        - Users can iterate on the theme-finding process if needed.
        """)

    st.divider()

    st.subheader(":green[Phase 5 - Defining and Naming Themes]")
    
    with st.expander("Braun and Clarke's Approach"):
        st.write("""
        This phase involves ongoing analysis to refine the specifics of each theme and 
        the overall story the analysis tells, generating clear definitions and names for each theme.
        """)

    with st.expander("Our App's Approach"):
        st.write("""
        - The app generates theme names and descriptions automatically.
        - Users can review and refine these theme names and descriptions.
        - The Theme-Codes book provides a comprehensive view of themes, codes, and associated quotes for further refinement.
        """)

    st.divider()

    st.subheader(":green[Phase 6 - Producing the Report]")
    
    with st.expander("Braun and Clarke's Approach"):
        st.write("""
        This final phase involves selecting vivid, compelling extract examples, final analysis 
        of selected extracts, relating back of the analysis to the research question and literature, 
        and producing a scholarly report of the analysis.
        """)

    with st.expander("Our App's Approach"):
        st.write("""
        - While the app doesn't write the final report, it provides several tools to aid in this process:
          - The Theme-Codes book provides a structured overview of themes, codes, and relevant quotes.
          - Visualizations like the Icicle and Network Map can be used to illustrate the thematic structure and relationships.
          - The Saturation Metric can be used to support methodological discussions.
        
        Note: The report writing itself still requires the researcher's expertise to interpret the results, 
        relate them to research questions and existing literature, and craft a coherent narrative.
        """)

    st.divider()

    st.subheader(":orange[Additional Features]")

    st.write("""
    1. **Saturation Metric:** Our app includes a tool to measure Initial Thematic Saturation (ITS), 
       which can help researchers assess the comprehensiveness of their analysis.

    2. **Visualizations:** The Theme-Codes Icicle and Thematic Network Map provide visual representations 
       of the thematic structure, aiding in the review and interpretation of results.

    3. **Flexibility:** Throughout the process, users can adjust prompts, model settings, and iterate 
       on different stages to refine their analysis.
    """)

    st.info("""
    Remember, while our app provides powerful tools for thematic analysis, it's designed to augment 
    rather than replace the researcher's expertise. Critical thinking, contextual understanding, 
    and interpretative skills remain crucial throughout the analysis process.
    """)

    # Call API key management function
    manage_api_keys()

if __name__ == "__main__":
    main()