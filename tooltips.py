'''
Tooltips for on-the-fly instructions

'''

project_tooltip = 'Select the project to use. For project set up and file management, see the 🏠 :orange[Project Set Up] page'

model_temp_tooltip = 'Model temperature controls how deterministic the large language model (LLM) will be. Lower temperatures yield more deterministic outputs, higher temperatures encourage more creativity. For replicable analyses, we recommend setting a low temperature.'

top_p_tooltip = 'Top_p, or nucleus sampling, is a setting that decides how many words to consider. A high top_p value means the model looks at more possible words, even less likely ones, making the generated output more diverse. As with model temperature, a low setting will increase the replicability of analysis.'

prompt_tooltip = 'This is the prompt that the LLM will receive alongside your transcripts. You can edit the prompt directly by changing the text, but you should ensure to retain the instructions relating to the formatting of the output.'

presets_tooltip = 'Select from a range of preset prompts for a range of typical analyses; you can create and manage your own preset prompts from the 📢 :orange[Prompt Settings] page'

model_tooltip = 'Select which large language model (LLM) you would like to perform this stage of the analysis. To use models, ensure you have the correct API keys set :orange[in the side bar]'