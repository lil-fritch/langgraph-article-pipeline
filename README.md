# Article Pipeline

A LangGraph-orchestrated pipeline that turns a topic into a fact-checked,
RAG-grounded article: generate search queries, search the web, validate
facts against local ground truth, outline, write section-by-section with
retrieval-augmented context, compile to Markdown.

> This is a generalized, from-scratch reimplementation of a content pipeline
> architecture I built professionally for a product in a different domain.
> The topic, prompts and example data here are generic; no original client
> data or business logic is included.

## Why a graph, not a single big prompt

A single "write me an article about X" prompt hallucinates confidently.
Splitting generation into nodes with their own narrow job makes each step
cheap to debug and lets facts flow forward without being re-invented by a
later step:

```
local_facts_check ──▶ query_generator ──▶ broad_search ──▶ fact_validator
                                                                  │
                                                                  ▼
       compiler ◀── section_writer ◀── scraper_indexer ◀── strategist
```

- **local_facts_check** — optional shortcut: if the topic matches a row in
  `data/topic_facts.example.csv`, those fields become non-negotiable ground
  truth for every later step (see `src/topic_facts.py`).
- **query_generator** — turns the topic (+ known facts) into 3-5 targeted
  search queries, explicitly told to search only for what's *missing*.
- **broad_search** — runs the queries against a SearXNG instance in
  parallel, dedupes by URL.
- **fact_validator** — the anti-hallucination gate. Classifies whether the
  topic refers to something real, whether the requested feature actually
  exists, and picks a writing strategy (`DIRECT_REVIEW`, `MYTH_BUSTER`,
  `CATEGORY_OVERVIEW`, ...) based on that classification. Produces an
  `ArticlePassport` that every later node treats as the single source of
  truth — a `MYTH_BUSTER` pivot because a feature is missing is a decision
  made once, not re-derived per section.
- **strategist** — turns the passport + search results into a section-by-
  section outline, enforcing that each section covers a distinct angle
  (no repeating the same metric across sections).
- **scraper_indexer** — scrapes the search results, strips cookie-banner /
  bot-check noise, chunks by Markdown headers, embeds via the OpenAI API,
  and indexes into an ephemeral per-topic LanceDB table (deleted after the
  article is compiled).
- **section_writer** — writes each section with RAG context from the
  indexer, under a "one truth policy" that pins it to the passport's
  verified facts and rejects invented brand names / competitor mentions.
- **compiler** — assembles sections in outline order, writes the Markdown
  file, and triggers RAG cleanup.

## Stack

Python 3.11+, LangGraph, OpenAI API (chat + embeddings + image generation —
cloud, not a local model server), LanceDB (embedded vector store),
trafilatura (scraping), SearXNG (web search backend).

## Usage

```bash
pip install -r requirements.txt
cp env.example .env   # set OPENAI_API_KEY and SEARCHXNG_HOST

python main.py --topics topics.example.txt --limit 3
python main.py --topics topics.example.txt --with-cover   # + DALL-E cover image
```

Topics are just a text file, one per line (or a CSV with a `topic`
column) — see `src/utils/topic_loader.py`. No keyword-expansion system
required; drop in your own list.

Generated articles land in `data/output/articles/*.md`.

## Swapping the domain

Everything that assumes "product review" lives in the prompts inside
`src/production/nodes/{query_generator,fact_validator,strategist,writer}.py`
and the `TopicSpecs` shape in `src/topic_facts.py`. The graph, the RAG
pipeline, and the LLM client don't know or care what the topic is about.
