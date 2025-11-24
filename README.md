# RAG Retrieval Pipeline Challenge

Build an AI agent to find similarly results base on user input text
The data set to test is WordPress QA dataset from hugging face: https://huggingface.co/datasets/mteb/qadupstack-wordpress


## Repository Structure
```
.
├── Makefile                  
├── docker-compose.yml        
├── pyproject.toml            
├── src/agentic_rag/          
└── scripts/download_dataset.py
```

## Configuration

Specify your implementations via environment variables or `.env`:
```bash

export AGENTIC_RAG_INGESTION_CLASS=agentic_rag.data.wordpress.WordPressPipLine
export AGENTIC_RAG_AGENT_CONTROLLER_CLASS=agentic_rag.agent.controller.WordPressAgentController
export AGENTIC_RAG_EVALUATOR_CLASS=agentic_rag.evaluation.runner.WordPressEvaluator
export AGENTIC_RAG_DATABASE__USER=db_user
export AGENTIC_RAG_DATABASE__PASSWORD=db_pass
export AGENTIC_RAG_DATABASE__DBNAME=db_name
export AGENTIC_RAG_DATABASE_URL=postgresql+psycopg2://db_user:db_pass@localhost:5432/db_name
```


Reorganize as needed, but preserve the CLI commands for reproducibility.

## Setup and Execution
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make compose-up        
make data              
make ingest            
make agent             
```

Stop the pgvector service with `make compose-down`. If the dataset requires authentication, export `HF_TOKEN` before downloading.
![Swager](images/swager.png)

# Test the API:
Got to http://localhost:8000/docs 
Execute query with /chat


# To evaluate the model:
```bash
make eval
```
or run python cmd:
```python
 python -m agentic_rag.cli evaluate  
```



