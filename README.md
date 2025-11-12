# Palmer Penguins — Aplicação Web com Análise de Dados e Machine Learning

Uma aplicação interativa em Python com **Flask** para análise de dados e modelos de Machine Learning. Desenvolvida para a disciplina Tópicos Especiais em Software.

---

## FUNCIONALIDADES PRINCIPAIS

- **Upload de Dados:** Carregue arquivos `.csv` ou use o dataset padrão _Palmer Penguins_.
- **Visualização Interativa:** Gráficos dinâmicos com `Plotly` (barras e dispersão) renderizados no template.
- **Seleção de Modelos:** Escolha entre `Decision Tree`, `KNN`, `Regressão Logística` e `SVM`.
- **Hiperparâmetros Ajustáveis:** Modifique parâmetros como profundidade da árvore ou número de vizinhos `K` via formulário.
- **Treinamento e Avaliação:** Execute o modelo e visualize a acurácia na página após o treino.
- **Predição Interativa:** Insira novos dados em um formulário e veja a espécie prevista após o envio.

---

## TECNOLOGIAS UTILIZADAS

- **Python 3.x:** Linguagem principal
- **Flask:** Backend web, roteamento e renderização de templates
- **Pandas:** Manipulação e análise de dados
- **Scikit-learn:** Modelos e métricas de Machine Learning
- **Joblib:** Para salvar e carregar o modelo treinado no disco
- **Plotly:** Criação de gráficos interativos (convertidos para HTML)
- **Seaborn:** Usado para o carregamento do dataset padrão

---

## INSTALAÇÃO E EXECUÇÃO

### 1. Pré-requisitos

Certifique-se de ter o [Python 3.7+](https://www.python.org/downloads/) instalado em seu sistema.

### 2. Criação do Ambiente Virtual

Abra um terminal na pasta do projeto e execute:

python -m venv venv

### 3. Ativação do Ambiente Virtual

Windows (PowerShell):
.\venv\Scripts\activate

macOS/Linux:
source venv/bin/activate

### 4. Instalação das Dependências
Com o ambiente ativado, instale todas as bibliotecas necessárias:

pip install flask pandas seaborn plotly scikit-learn matplotlib joblibb

### 5. Execução da Aplicação
Finalmente, execute a aplicação:

python app.py


O terminal indicará que o servidor está rodando em um link (ex: http://127.0.0.1:5000/). Abra este link no seu navegador para usar a aplicação.

--------------------------------------------------

Observações Importantes

* Utiliza o dataset Palmer Penguins, uma alternativa moderna à base Iris.
* O código é modular e expansível — novos modelos e gráficos podem ser adicionados facilmente.
* Todas as análises e interações são feitas através de formulários HTML e rotas do Flask, que atualizam a página com os resultados.

--------------------------------------------------

Projeto acadêmico desenvolvido para fins educacionais
```
