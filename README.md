## How fast is `llm_classify`?

Benchmarking Phoenix Arize [`llm_classify` utility](https://docs.arize.com/phoenix/api/evals#phoenix.evals.llm_classify) 

## Setup
> pip intsall -r requirements.txt


## Run Benchmarks

### Phoenix benchmark w/ `llm_classify`

```bash
python phx.py
```
> Time for llm_classify: 192.21098494529724

### Hamel's handrolled thing with `threadpool`

```bash
python hamel.py
```
> Time for hamel's script: 63.86933922767639
