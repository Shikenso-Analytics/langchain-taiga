# langchain-taiga-shikenso

This package contains the LangChain integration with TaigaShikenso

## Installation

```bash
pip install -U langchain-taiga-shikenso
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatTaigaShikenso` class exposes chat models from TaigaShikenso.

```python
from langchain_taiga_shikenso import ChatTaigaShikenso

llm = ChatTaigaShikenso()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`TaigaShikensoEmbeddings` class exposes embeddings from TaigaShikenso.

```python
from langchain_taiga_shikenso import TaigaShikensoEmbeddings

embeddings = TaigaShikensoEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`TaigaShikensoLLM` class exposes LLMs from TaigaShikenso.

```python
from langchain_taiga_shikenso import TaigaShikensoLLM

llm = TaigaShikensoLLM()
llm.invoke("The meaning of life is")
```
