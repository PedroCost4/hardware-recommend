
## What is this?

This is a project to crawl websites and recommend hardware products based on a prompt from a user.


## Installation
- Clone the repository
- Install python (In this case, using homebrew)
- Install [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/pedrocosta/hardware-recommend.git
cd hardware-recommend
brew install python 
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Install dependencies

```bash
uv pip install -r pyproject.toml
cp .env.example .env
```

## Configuration

- Create a `.env` file in the root directory with the following content:
 - INDEX_PATH: The path where the index will be stored
 - EMBEDDING_MODEL: The embedding model to use
 - local_llm: If true, will use the local LLM, if false, will use the cloud LLM
 - if cloud LLM, you need to set the following environment variables:
   - API_KEY according to llama index documentation
 - to run cron job, you need to set the following environment variables:
   - TESTING: If true, will run the crawler every 5 minutes, if false, will run it every day at 8:00 AM

## Excution

```bash
uv run main.py
```

## Cron

- This cron job will run everyday at 8:00 AM to crawl the websites and update the database.

You can make this run in a VPS to get the sites crawled everyday.

```bash
uv run cron.py
```