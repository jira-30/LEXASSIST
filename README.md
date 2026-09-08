# LEXASSIST - (Automating Legal Intelligence: Design and Implementation of a Multi-Agent LLM System for Legal Clause Analysis and Risk Detection)

**Author:** Shruthi Ravi

**Dataset**: The Legal-Clause -> 395 legal datasets (Kaggle) 

**Framework**: Supervisor coordinated multi agent system built through LangGraph, FastAPI and Streamlit with retrieval grounded in ChromaDB vector database. 

**Task:** **Following are the list of situations faced by people through legal lens every single day**
```
Scenario 1 -> A startup CEO is about to sit down with an investor to negotiate a term sheet.
Her legal advisor is out sick and unreachable.
She isn't trained to read clauses like liquidation preference or indemnification,
and has no way to tell whether what's on the page is standard or unusually aggressive.
```
```
Scenario 2 -> An international student misses a rent payment after an unexpected financial setback.
His landlord sends a notice threatening legal action.
The student has never read a lease closely, doesn't know what a "default clause" or "cure period" means,
can't afford a lawyer, and is now worried this could spiral into a court case or affect his visa status.
```
As above mentioned scenarios, legal documents such as contracts, policies and amendments are often long, complex and difficult to interpret in real time, especially under time pressure. Existing legal resources are not designed for instant querying, structured understanding or risk evaluation. Sometimes these scenes require high trained professional lawyers leading to financial constraints. By considering all the above situations, LEXASSIT was developed for consumers to close that gap. It instantly searches, analyses and summarizes the real world clauses, enabling faster, more informed decisions without requiring legal assistance.

## What It Does?
+ **Classifies** the document or query, figures out the kind of document or legal clause it is spread across the dataset.
+ **Retrieves** relevant context by pulling out the similar clauses and reference material from ChromaDB vector store built with datasets as the base leading through solutions match the actual data.
+ **Assesses** **risk** by flagging whether a clause is unusual, aggressive or questionable.
+ **Summarizes** reproduces the whole document into simple terms covering its whole information, obligations, risk and options to use as per consumer's needs.

## Architecture

#### PHASE 1 
![LexAssist Architecture](LexAssist_Phase1.png)

