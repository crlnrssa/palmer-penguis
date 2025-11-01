Palmer Penguins — Aplicação Web com Análise de Dados e Machine Learning

Uma aplicação interativa em Python com Streamlit para análise de dados e modelos de Machine Learning. Desenvolvida para a disciplina Tópicos Especiais em Software.

--------------------------------------------------

FUNCIONALIDADES PRINCIPAIS

* Upload de Dados: Carregue arquivos .csv ou use o dataset padrão Palmer Penguins.
* Visualização Interativa: Gráficos dinâmicos com Plotly (barras, pizza e dispersão).
* Seleção de Modelos: Escolha entre Decision Tree, KNN, Regressão Logística e SVM.
* Hiperparâmetros Ajustáveis: Modifique parâmetros como profundidade da árvore ou número de vizinhos K.
* Treinamento e Avaliação: Execute o modelo e visualize acurácia e matriz de confusão em tempo real.
* Predição Interativa: Insira novos dados e veja a espécie prevista instantaneamente.

--------------------------------------------------

TECNOLOGIAS UTILIZADAS

* Python 3.x: Linguagem principal
* Streamlit: Interface web interativa
* Pandas: Manipulação e análise de dados
* Scikit-learn: Modelos e métricas de Machine Learning
* Plotly: Criação de gráficos interativos
* Matplotlib / Seaborn: Visualizações complementares

--------------------------------------------------

INSTALAÇÃO E EXECUÇÃO

1. Pré-requisitos
Certifique-se de ter o Python 3.7+ instalado em seu sistema.

2. Criação do Ambiente Virtual
Abra um terminal na pasta do projeto e execute:

python -m venv venv

3. Ativação do Ambiente Virtual

Windows (PowerShell):
.\venv\Scripts\activate

macOS/Linux:
source venv/bin/activate

4. Instalação das Dependências
Com o ambiente ativado, instale todas as bibliotecas necessárias:

pip install streamlit pandas seaborn plotly scikit-learn matplotlib

5. Execução da Aplicação
Finalmente, execute a aplicação:

streamlit run app.py


A aplicação abrirá automaticamente em uma nova aba do navegador, permitindo explorar os dados e treinar modelos interativamente.

--------------------------------------------------

Observações Importantes

* Utiliza o dataset Palmer Penguins, uma alternativa moderna à base Iris.
* O código é modular e expansível — novos modelos e gráficos podem ser adicionados facilmente.
* Todas as análises e interações são feitas diretamente na interface Streamlit, sem alterar código manualmente.

--------------------------------------------------

Projeto acadêmico desenvolvido para fins educacionais
