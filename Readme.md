## Deterministic Candidate Evaluation
# Read: Omar's Solution Slides

Picks a random Candidate, match with the approperiate job using similarity search (RAG), run an evaluating LLM 10 times

# How to run?
```
python workflow.py
```

# Previous results
- Look at the folders logs_runX, each folder has 10 json files of the result of each run
- Look at results_RAG.json for the matched jobs


~Omar