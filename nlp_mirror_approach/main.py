import os
from apikey import apikey
os.environ["OPENAI_API_KEY"] = apikey


# importing Langchain libraries
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain,SequentialChain
from langchain.agents import create_csv_agent

llm_noncreative = OpenAI(temperature=0.2)
llm_creative = OpenAI(temperature=0.6)

# Define the user input

# Positive Sentiment

# Negative Sentiment
# In a quiet corner of a forgotten town, a melancholic unicorn named Ash and a somber dog named Ember found solace in each other's company. Ash, with their muted gray coat and a worn-out horn, carried the weight of past regrets. Ember, a subdued and introspective pup, had endured the hardships of a difficult life. Their paths converged on a misty evening, forming an understanding born of shared sorrow. Ash possessed no magical powers but harbored a gentle empathy that soothed Ember's wounded spirit. Together, they traversed the dimly lit streets, offering comfort to those plagued by loneliness, painting their melancholy with a touch of understanding.

UserInput = """   

India strongly condemned the European Parliament's adoption of a resolution regarding recent clashes in Manipur, asserting that such interference in India's internal affairs is unacceptable. The external affairs ministry spokesperson emphasised that the Indian authorities, including the judiciary, are fully aware of and addressing the situation in Manipur to ensure peace, harmony, and law and order. India advised the European Parliament to focus on its internal matters instead. Earlier, the foreign secretary clarified that the issue is entirely internal to India and expressed that this point was made clear to the concerned EU parliamentarians. The resolution passed by the European Parliament urged Indian authorities to halt ethnic and religious violence, protect religious minorities, conduct independent investigations, and encourage conflicting parties to rebuild trust and play an impartial role in mediating tensions. Notably, Prime Minister Narendra Modi embarked on a visit to France for significant meetings with French President Emmanuel Macron and other leaders on the same day the resolution was adopted.

"""

# first, the outputs of each steps were visible for demonstration, but now i chained two steps in one prompt for efficiency and request limits for the API. I now have made block system, so it does not hit the limit

# Preprocessing and Parameter Extraction Chain - Block_1

final_prompt = PromptTemplate(
    input_variables=["UserInput"], 
    template= """ 

    ``` UserInput : {UserInput} ```

    The input is given to you above delimited by ``` ```. Clean the data and Extract the following parameters from the data ranging in the value -1 to 1. Perform basic analysis using your capabilities and do this independently for all parameters. Don't output the parameter name and score, but remember them. The parameters are explained below : 

    1. Sentiment Score
    2. Subjectivity Score
    3. Concreteness Score
    4. Polarity Score
    5. Arousal Scoret

    Perform the second step three times internally and then, perform this step next and give the output. After this, using the scores you remember, find top 5 similar words from the csv dictionary below. Don't output the parameter scores.

    ``` These are Strict Instructions ```

    Output should be in the following format :
    
    ```Wordlist that you found after searching from the below csv dictionary```

    ``` Summarize the input text and rephrase the new summary using the above 5 words in the wordlist. It should not look like a summary though. So, prevent that type of indentifiers. Wrap them inside ** whenever used. Use them strictly, try to form meaning. ```


    Dictionary in csv format : 

    Word,Sentiment Score,Subjectivity Score,Concreteness Score,Polarity Score,Arousal Score

    Effervescent,0.7,0.5,0.4,0.6,0.8
    Enigmatic,0.6,0.6,0.5,0.2,0.4
    Mellifluous,0.7,0.3,0.6,0.5,0.4
    Surreptitious,0.4,0.5,0.5,0.3,0.3
    Ephemeral,0.3,0.4,0.3,0.2,0.4
    Clandestine,0.2,0.5,0.4,0.1,0.3
    Serendipity,0.8,0.7,0.5,0.8,0.9
    Quintessential,0.6,0.3,0.7,0.4,0.3
    Ethereal,0.7,0.5,0.4,0.6,0.7
    Opulent,0.7,0.4,0.7,0.6,0.6
    Incandescent,0.6,0.4,0.5,0.5,0.7
    Ineffable,0.5,0.8,0.3,0.1,0.6
    Serene,0.7,0.3,0.7,0.6,0.4
    Resonance,0.5,0.4,0.6,0.2,0.5
    Serenity,0.8,0.4,0.6,0.7,0.2
    Euphoria,0.8,0.6,0.8,0.7,0.9
    Querulous,0.2,0.7,0.4,0.1,0.4
    Luminary,0.7,0.4,0.6,0.5,0.8
    Tranquil,0.7,0.4,0.7,0.6,0.5

    """
)


# Defining chains and inputs for block_1

final_chain = LLMChain(llm=llm_noncreative, prompt=final_prompt)
final_rephrase = final_chain.run(UserInput) # final parameter scores for input paragraphs
print(final_rephrase)
