import tiktoken

'''
encoding = tiktoken.get_encoding("o200k_base")
tokens = encoding.encode("holysmokesbatman")
print(tokens)
print(len(tokens))
'''

def approx_tokens_count(message):
    """ Estimate the number of tokens ina  given prompt , using o200k_base encoding"""

    encoding = tiktoken.get_encoding("o200k_base")
    tokens = encoding.encode(message)
    num_tokens = len(tokens)
    return num_tokens
