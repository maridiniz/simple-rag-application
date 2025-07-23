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
O projeto tem o intuito apenas de demonstrar o passo a passo de uma simples aplicação RAG (Retrieval-Augmented Generation). São utilizados os components dos frameworks LangChain, langGraph e alguns components integrados como Google GenAI para os passos de vetorização e generação do output, assim como também o Chroma client para indexição dos embeddings localmente (no caso deste projeto) e posteriormente o passo de recuperação. Também é possível utilizar ainda em conjunto outros frameworks para observabilidade da nossa aplicação, como por example o LangSmith, onde é possível monitorar, realizar debug e etc. Porém, para este projeto, este último passo não será abordado, caso tenha intersse, veja como integrar este passo na aplicação, acesse [aqui](https://docs.smith.langchain.com/observability). É possível encontrar todas as informações diretamenta na documentação official do LangChain neste [link](https://python.langchain.com/docs/introduction/). Quanto a documentação official do LangGraph, acesse [aqui.](https://langchain-ai.github.io/langgraph/)

O RAG é uma aplicação no modelo de perguntas e respostas, onde é realizada uma busca nos nossos próprios arquivos, ao final, tanto o prompt quanto o conteúdo recuperado são passados ao modelo de llm escolhido para gerar um output mais amigável. 


# Setup
Para este projeto serão necessárias as sequintes dependências:
- `langchain-google-genai >= 2.1.6 `
- `langchain-chroma >= 0.2.4`
- `langchain-community >= 0.3.27`
- `langchain-text-splitters >= 0.3.8`
- `langchain-core >= 0.3.68`
- `langchain >= 0.3.26`

Dependências built-in:
- `os`
- `typing_extensions`

Para  realizar a instalação das dependências citadas acima, execute no terminal o comando:
```{python}
pip install langchain-google-genai langchain-chroma langchain-community langchain-text-splitters langchain-core langchain
```

Para obter uma cópia deste repositório, execute o comando:
```bash
git clone git@github.com:maridiniz/simple-rag-application.git
```


Para prosseguir com o projeto é necessário utilizar um LLM que fará o processo de vetorização, indexação e também participará do processo de gerar o output para o usuário, por tanto,  será necessária utilizar uma chave de API de algum modelo de LLM ou utilizar um modelo local. Para definir a chave de API do modelo LLM escolhido em uma variável de ambiente:

```
1. opção através do powersehll (Método persiste mesmo após reiniciar o sistema):
[System.Environment]::SetEnvironmentVariable('GOOGLE_API_KEY', 'insira aqui sua chave', 'User')
```

```
2. opção ataravés do CMD (Método permance mesmo após reiniciar):
setx GOOGLE_API_KEY "insira-aqui-sua-chave"
```

Para verificar se funcionou:
```
powershell: echo $env:GOOGLE_API_KEY

cmd: echo %GOOGLE_API_KEY%
```

Outra alternativa com o módulo getpass:
```
import getpass
import os

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Insira sua chave de API aqui: ")
```


# Estrutura do projeto
A aplicação segue cinco passos:

1. O primeiro passo é uma função que define toda a lógica para o processamento dos arquivos que serão futuramente vetorizados e indexados em um diretório local. Primeiro, os arquivos são carregados do diretório onde se encontram, logo depois são contidos em um único objeto como uma unidade de texto. Ao final, essa unidade de texto é dividida em várias partes menores para que sejam futuramente transformados em embeddings.
![](/image/data_processing.png)

2. O segundo passo define todo o processo de vetorização e indexação dos arquivos previamente carregados e divididos no primeiro passo. Nessa etapa é utilizado um modelo de llm para transformar cada pedaço de texto em embedding, um processo de atribuição de um valor numérico para cada pedaço de texto. Posteriormente, ocorre a indexação desses embeddings, ou seja, o armazenamento em banco de dados local (Nesse caso, foi utilizado o Chroma).
![](/image/indexing_embedding.png)

3. O terceiro passo define o processo de busca com base nas perguntas feitas pelo usuário. É realizada uma recuperação dos arquivos indexados no diretório escolhido no passo 2, e então esse resultado, chamado de contexto é passado para o quarto passo para gerar a resposta final ao usuário através de um modelo de linguagem.
![](/image/retieve_generation.png)

4. O quarto passo é onde definimos o prompt que será passada ao llm, onde estará tanto a pergunta do usuário quanto o contexto (o conteúdo recuperado dos nossos arquivos) para que seja gerado o output.

5. O quinto e último passo é onde agrupamos todos os passos anteriores da nossa aplicação com o LangGraph.

# Demo
![](/image/rag_app_video.mp4)
