## ollama pull mistral
## ollama pull nomic-embed-text

## (alternatives plus puissantes -> mxbai-embed-large ou snowflake-arctic-embed / dim = 1024)

## pip install -r requirements.txt

## uvicron mian:app --relaod


## CURL

curl -X POST -F "file@path" http://localhost:8000/documents/upload

curl -X POST http://loclahost:8000/query \
-H "Content-Type": application/json"\
-d '{"question": "parle moi du document}'# rag_worker
