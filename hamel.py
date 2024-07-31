from openai import OpenAI
import pandas as pd
from fastcore.parallel import parallel
import time
client = OpenAI()

SYS_PROMPT="""You are a real estate agent using software that assists you.
You are presented with a question and an answer.  You need to judge if the answer is helpful or not. 
Respond with either "True" or "False" depending on if the answer is helpful or not."""

CATEGORICAL_TEMPLATE = """
Here is the question and answer:

<question>
{question}
</question>

<answer>
{answer}
<answer>
"""

def llm_classify(question,answer):
    response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": CATEGORICAL_TEMPLATE.format(question=question, answer=answer)},
      ],
        max_tokens=1,
        logit_bias={2575:1, 3641:1}
    )
    return response.choices[0].message.content

def parallel_classify(d): 
    return llm_classify(**d)

if __name__ == '__main__':
    df = pd.read_csv('data.csv')[['question', 'answer']].to_dict(orient='records')
    start = time.time()
    results = parallel(parallel_classify, df, threadpool=True, progress=True, total=len(df), n_workers=16)
    print(f"Time for hamel's script: {time.time() - start}")