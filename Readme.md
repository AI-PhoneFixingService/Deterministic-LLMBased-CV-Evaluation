## Deterministic Candidate Evaluation
### Read: OmarSolutionSlides.pdf - for more details understand the solution
### Decompress results.zpi - for results of 10 llms calls on the same Candidate & Job
### Open results_RAG.json - for results of Job matching based on the candidate


## Script steps
- Parses the data into different json files : jobs_only.json & candidates.json
- Picks a random Candidat
- Matchs the Candidate with an approperiate job using similarity search (RAG)
- Runs an evaluating LLM 10 times


## How to run?
```
python workflow.py
```

## Previous results
- Decompress results.zip, then look at the folders logs_runX, each folder has 10 json files of the result of each run (Deterministic LLM results)
- Look at results_RAG.json for the matched jobs (RAG)


## Notes on the Results
- The solution is designed to minimize the error margine (outputs almost identical) BUT with the risk of high varient outliers (LLM hallucination 10-15% chance)
- For Production: Run odd number of calls and use the Mode value as the output (the most frequent - less probably to be the hallucintaed output- )

~Omar
