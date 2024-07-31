import os
from phoenix.evals import OpenAIModel, llm_classify
import pandas as pd
import time

CATEGORICAL_TEMPLATE = """You are a real estate agent using software that assists you.
You are presented with a question and an answer.  You need to judge if the answer is helpful or not. 
Respond with either "True" or "False" depending on if the answer is helpful or not.

<question>
{question}
</question>

<answer>
{answer}
<answer>
"""

eval_model = OpenAIModel(model="gpt-4o", api_key=os.getenv('OPENAI_API_KEY'))
df = pd.read_csv('data.csv')

rails = ["True", "False"]

start = time.time()
relevance_classifications = llm_classify(
    provide_explanation=False,
    dataframe=df,
    template=CATEGORICAL_TEMPLATE,
    model=eval_model,
    rails=rails,
    concurrency=16)

print(f"Time for llm_classify: {time.time() - start}")
