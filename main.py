import openai
import streamlit as st
import os
openai.api_key = st.secrets["pass"]

st.set_page_config(page_title="Reword : Learn New Shit !")
st.header("Reword Prompt Preview !")


wordlist = st.text_area("Enter comma seperated word list : ")
input_para = st.text_area("Enter the paragraph to reword : ")




prompt_system = f""" 

Your are working as a paraphrasing tool. Your job is to take as input a paragraph, and reword it using new word that are provided to you below in a list. Keep the sentiment, tone and meaning of the text same while rewording the paragraph. Check out for idioms, sayings and quotes as you don't need to touch them, keep them as they are. The new text that is output has the purpose of teaching new words to the readers, so that they can know the application of words in real-life context. Wrap new words used around **. Don't use any words outside the words provided to replace. Try to use all the words from the wordlist. Also, the output should only contain the paragraph, don't output anything unnecessary.

******

Some Examples only for reference : 

Original: Hulu commands a sizeable market.
Rewrite: (wordlist : wields, reach) : Hulu wields a vast reach.

Original: Milk strengthens one's bones.
Rewrite: (wordlist : documented, bolster) : Milk has been documented to bolster one's bone strength.

Original: GMOs command a negative reputation, but this sentiment is wrong.
Rewrite: (wordlist : tainted, inaccurate) : GMOs bear a tainted legacy, but this assessment is inaccurate.

******

Expect a word-list and a paragraph in the user-prompt. Don't output anything other than the required paraphrased output.

"""


prompt_user = f""" 

Your Output : 
Original : {input_para}
Rewrite: (wordlist : {wordlist}) : ``` Insert Output Here ```

"""

def code_gen(temperature=0.5):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ],
    )
    return response.choices[0].message.content


if st.button("REWORD ! ") :
    with st.spinner("Generating"):
        response = code_gen()
        st.success("Done !")

    st.text_area("Organized Dump !", response, height=400)


