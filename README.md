# Aplicação RAG Simples
[![LangChain](https://img.shields.io/badge/LangChain-4CAF50?style=flat-square&logo=langchain&logoColor=black)](https://python.langchain.com)
[![PyPI version](https://badge.fury.io/py/langchain.svg)](https://pypi.org/project/langchain/)
[![Downloads](https://static.pepy.tech/badge/langchain)](https://pepy.tech/project/langchain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/LangGraph-9E9E9E?style=flat-square&logo=python&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![PyPI version](https://badge.fury.io/py/langgraph.svg)](https://pypi.org/project/langgraph/)
[![Downloads](https://static.pepy.tech/badge/langgraph)](https://pepy.tech/project/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# Objetivo do projeto
O projeto tem o intuito apenas de demonstrar o passo a passo de uma simples aplicação RAG (Retrieval-Augmented Generation). São utilizados os components dos frameworks LangChain, langGraph e alguns components integrados como Google GenAI para os passos de vectorização e generation, assim como também o Chroma client para indexição dos embeddings localmente (no caso deste projeto) e posteriormente o passo de retrieval. Também é possível utilizar ainda em conjunto outros frameworks para observabilidade da nossa aplicação, como por example o LangSmith, onde é possível monitorar, realizar debug e etc. Porém, para este projeto, este último passo não será abordado, caso tenha intersse, veja como integrar este passo no aplicação, acesse [aqui](https://docs.smith.langchain.com/observability). É possível encontrar todas as informações diretamenta na documentação official do LangChain neste [link](https://python.langchain.com/docs/introduction/). Quanto a documentação official do LangGraph, acesse [aqui.](https://langchain-ai.github.io/langgraph/)

O RAG é uma aplicação no modelo de perguntas e respostas, onde é realizada uma busca nos nossos próprios arquivos, ao final, tanto o prompt quanto o conteúdo recuperado são passados ao modelo de llm escolhido para gerar um output mais amigável. 


# Setup
Para este projeto serão necessários os sequintes components instalados:
- `langchain-google-genai >= 2.1.6 `
- `langchain-chroma >= 0.2.4`
- `langchain-community >= 0.3.27`
- `langchain-text-splitters >= 0.3.8`
- `langchain-core >= 0.3.68`
- `langchain >= 0.3.26`

Para  realizar a instalação dos components citados acima, execute no terminal o comando:
```{python}
pip install langchain-google-genai langchain-chroma langchain-community langchain-text-splitters langchain-core langchain
```

Outros components built-in:
- `Python >= 3.12`
- `os`
- `typing_extensions`

Para obter uma cópia deste repositório execute o comando:
```bash
git clone git@github.com:maridiniz/simple-rag-application.git
```

Método alternativo em casos de HTTP:
```bash
git clone https://github.com/maridiniz/simple-rag-application.git
```

# Estrutura do projeto
A aplicação segue 5 (cinco) passos simples:
- O primeiro passo realiza o processamento dos arquivos que desejamos obter respostas.
- O segundo passo se encarrega de realizar a vetorização dos textos destes documentos e armazenamento dos vetores/embeddings em um banco de dados, que neste caso é local.
- O terceiro passo organiza a parte de recuperação dos textos armazenados ataravés de uma busca por similaridade.
- O quarto passo trata de gerar uma resposta ao usuário (gerado pelo llm) com base na pergunta e no contexto.
- O quinto e último passo trata de como unir todos esses quatro passos da nossa aplicação com LangGraph.
