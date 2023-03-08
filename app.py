"""
# PDF to SOP creation using LLM APIs - Tester App

# ToDOs:
# - cache the results of content & embedding so it doesn't get generated again 
# * Breakout and fix file uploader so its cleaner [here](https://levelup.gitconnected.com/python-streamlit-uploading-and-using-files-cf797dc30be3)
"""

import streamlit as st
import streamlit_ext as ste
import pandas as pd
import tiktoken
import PyPDF2
import numpy as np
import time
import json
import openai
import pickle
import tenacity

openai.api_key = st.secrets["openai_key"]

########## MAIN UI Components ##########

# Main app title
st.title('PDF to SOP Tool')

# Starting options for user
with st.form(key="user_form"):
    # User options 
    option = st.selectbox(
        'Pick source to generate SOP from',
        ('Existing PDFs', 'Upload My Own'))

    # if option == 'Existing PDFs':
    # Selector to generate pre-created embeddings
    source_select = st.selectbox(
    'Select source',
    ('Panasonic microwave manuel', 'Placeholder - DO NOT SELECT'))
    # elif option == 'Upload My Own':
    # Uploader to take in user specified PDF file 
    uploaded_file = st.file_uploader("Choose a PDF file (if option is selected)")

    # Topic or question requested
    user_query = st.text_input("Your query", key="user_query")

    # Optional user options
    pg_start = st.number_input('Page start', min_value=1, value=1)
    pg_end = st.number_input('Page end', min_value=1, value=100)

    # submit button
    submit_button = st.form_submit_button(label='Start generation')

# Instructions in the sidebar
st.sidebar.markdown("""
    ## About
    Takes long manuals in PDF and allows Q&A on the custom content. 

    ## Cautions
    - Avoid uploading long PDFs multiple times for now due to API limitations (instead download the resulting 2 files and contact me!)
    - Limit pages numbers when testing (min. 5 pages)

    _personal use only for testing, using OPEN AI APIs_
    """)

########## HELPER FUNCTIONS ##########

@st.cache_data
def convert_df_to_csv(df, incl_index=True):
   return df.to_csv(index=incl_index).encode('utf-8')

def create_file_download_button(content_type, csv):
    return ste.download_button(
        label = "Download " + content_type,
        data = csv,
        file_name = content_type + ".csv",
        mime = "text/csv",
        # key = 'download-csv-' + content_type
    )


########## USER INPUTS ##########
if submit_button:
    print(option,
    source_select,
    uploaded_file,
    user_query,
    pg_start,
    pg_end)                  

QUERY = user_query

# Based on user input
if option == 'Existing PDFs':
    PDF_FILE = None
    # load preprocessed data and embedding files
    if source_select == 'Panasonic microwave manuel':
        EMBEDDING_FILE = 'data/0_microwave_embeddings.csv'
        PROCESSED_FILE = 'data/0_microwave_processed.csv'
else:
    # use uploaded PDF file 
    PDF_FILE = uploaded_file.name
    PROCESSED_FILE = ""
    EMBEDDING_FILE = ""

########## MAIN SCRIPT ##########

########## DEFAULTS ##########
# Defaults
EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "gpt-3.5-turbo"   # "text-davinci-003"
ENCODING = "cl100k_base"  # encoding for ChatGPT models

# Prompt defaults
PROMPT_HEADER = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
MAX_SECTION_LEN = 1500  # 2000 for context incl. rest of prompt, save 2000 for completion 

CHAT_COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 2000,
    "model": COMPLETIONS_MODEL,
}

BASE_MESSAGE = [{"role": "system", "content": "You are a kind helpful assistant"}]

########## PDF text extraction, preprocessing, embedding creation ##########

def get_pdf_range(start, end, max_pg):
    return range()

def extract_pdf(pdf_data):
  # For now, each page is treated as a separate section of content and the first 
  # sentence of the page is treated as the heading 
  pdf_reader = PyPDF2.PdfReader(pdf_data)

  # Extract the text content from the PDF
  headings = []
  contents = []
  pg_nums = []
  for page in range(max(0,pg_start-1), min(pg_end, len(pdf_reader.pages))):
      text_content = pdf_reader.pages[page].extract_text()
      headings.append(text_content.split('\n')[0])
      contents.append(text_content)
      pg_nums.append(page + 1)  # start at pg 1

  # Create a Pandas dataframe from the headings and content
  return pd.DataFrame({'heading': headings, 'content': contents, 'pg_number': pg_nums})

def num_tokens_from_string(string, encoding_name):
  """Returns the number of tokens in a text string."""
  encoding = tiktoken.get_encoding(encoding_name)
  num_tokens = len(encoding.encode(string))
  return num_tokens

def count_tokens(df):
  # encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
  return df.apply(lambda x: num_tokens_from_string(x.content, "cl100k_base"), axis=1)
  # pdf_sections.nlargest(n=3, columns='tokens')

def preprocess_pdf_data(pdf_file):
  pdf_sections = extract_pdf(pdf_file)
  # basic text processing - this can be improved later
  pdf_sections.replace('\n',' ', regex=True, inplace=True)           # replace new line characters with space
  pdf_sections.replace(r'^(\d+)', '', regex=True, inplace=True)      # remove any leading numeric characters
  pdf_sections["tokens"] = count_tokens(pdf_sections)
  pdf_sections.drop_duplicates(subset=["heading"], keep="first", inplace=True)
  return pdf_sections

# Create embeddings from PDF content

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def get_embedding(text):
    result = openai.Embedding.create(
      model=EMBEDDING_MODEL,
      input=text
    )
    # time.sleep(2)  # force sleep 2 seconds for now 
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }

def load_embeddings(fname):
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "heading"])
    return {
           (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

# run section if new PDF uploaded
if submit_button and PDF_FILE:
    # Create a PDF reader object
    st.write("**Processing uploaded PDF document...**")
    pdf_sections = preprocess_pdf_data(uploaded_file)    
    # set column index for future search 
    pdf_sections.set_index("heading", inplace=True)    

    # Save processed PDF
    st.write("**Processed PDF contents - please save**")
    st.write(pdf_sections.sample(3))
    st.write("total sections: ", pdf_sections.shape[0])
    pdf_content_csv = convert_df_to_csv(pdf_sections)
    create_file_download_button('pdf_content', pdf_content_csv)

    # compute embeddings
    document_embeddings = compute_doc_embeddings(pdf_sections)

    # saving embeddings for future load
    st.write("**Processed PDF embeddings - please save**")
    example_entry = list(document_embeddings.items())[0]
    st.write(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")
    embedding_csv = convert_df_to_csv(pd.DataFrame.from_dict(document_embeddings, orient="index"))
    create_file_download_button('pdf_embedding', embedding_csv)

# run section if using existing data
if submit_button and option == 'Existing PDFs':
    # load PDF preprocessed content 
    pdf_sections = pd.read_csv('data/0_microwave_processed.csv')                # TODO: add @st.cache_data later with embeddings
    # set column index for future search 
    pdf_sections.set_index("heading", inplace=True)  
    st.write("**Using existing PDF preprocessed content... example:**")
    st.write(pdf_sections.sample(3))
    # load embedding
    embeddings_raw = pd.read_csv(EMBEDDING_FILE)
    embeddings_raw.columns.values[0] = "heading"
    embeddings_raw.set_index("heading", inplace=True)
    document_embeddings = embeddings_raw.T.to_dict('list')
    st.write("**Using existing PDF preprocessed content... example:**")
    example_entry = list(document_embeddings.items())[0]
    st.write(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")
    embedding_csv = convert_df_to_csv(pd.DataFrame.from_dict(document_embeddings, orient="index"))

############### CREATE PROMPT AND SEND TO CHATGPT #########

# get context separators to make prompt easier to read
SEPARATOR = "\n* "
encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query, contexts):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

def construct_prompt(question, context_embeddings, df, diagnostic=False):
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space. may want to change this later       
        document_section = df.loc[section_index]
        # print(section_index, " ", document_section.tokens)
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    if diagnostic:
      print(f"Selected {len(chosen_sections)} document sections:")
      print("\n".join(chosen_sections_indexes))
    
    full_prompt = PROMPT_HEADER + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

    result = {"prompt" : full_prompt, "ref" : chosen_sections_indexes}
    
    return result

def create_reference(df, indexes):
  references = []
  for index in indexes:
    reference = "Page " + str(df.loc[index].pg_number) + ": " + index
    references.append(reference)
  return "\n\nReferenced from: " + ("\n").join(references)

def answer_query_with_context_chatgpt(
    query,
    df,
    document_embeddings,
    show_prompt=False,
    show_diagnostic=False,
    show_reference=True
):
    prompt = construct_prompt(
        query,
        document_embeddings,
        df,
        show_diagnostic
    )

    message = BASE_MESSAGE + [{"role":"user", "content":prompt["prompt"]}]

    if show_prompt:
        print(message)   

    if show_reference:
      reference_text = create_reference(df, prompt["ref"])
    else:
      reference_text = ""

    response = openai.ChatCompletion.create(
                messages=message,
                **CHAT_COMPLETIONS_API_PARAMS
            )

    full_response = response["choices"][0]["message"]["content"] + reference_text

    return {"response" : full_response, "query" : message}

if submit_button:

    # create and submit the query 
    st.write("**Creating and submitting query...**")
    result = answer_query_with_context_chatgpt(QUERY, pdf_sections, document_embeddings, True, True, True)

    st.write("**Resulting response...**")
    result["response"]

    with st.expander("See full prompt"):
        result["query"]
        