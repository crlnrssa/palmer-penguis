# --- 1. Importações ---
import streamlit as st  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  
import plotly.express as px  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, confusion_matrix  

# Importa as classes dos modelos de Machine Learning que vamos usar
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings  # Usado para ignorar mensagens de aviso

# Ignorar warnings futuros (ex: de versões de bibliotecas)
warnings.filterwarnings('ignore')

# --- 2. Configuração da Página ---
st.set_page_config(
    layout="wide",  # Define que a aplicação usará todo o espaço horizontal da tela
    page_title="Palmer Penguins App",  
)

# --- 3. Funções de Carregamento e ML ---

# O '@st.cache_data' é um "decorator" do Streamlit
# Ele diz ao Streamlit para "lembrar" o resultado desta função
# se a função for chamada de novo, ele retorna o resultado salvo (em cache)
# em vez de executar
@st.cache_data
def carregar_dados_padrao():
    """Carrega o dataset 'penguins' do Seaborn e faz uma limpeza básica."""
    try:
        penguins = sns.load_dataset('penguins')
        penguins = penguins.dropna()  # Remove linhas com valores nulos (NaN)
        return penguins
    except Exception as e:
        st.error(f"Erro ao carregar dados padrão: {e}")
        return pd.DataFrame()  # Retorna um DataFrame vazio em caso de erro

def carregar_dados_upload(arquivo_enviado):
    """Lê um arquivo CSV enviado pelo usuário."""
    if arquivo_enviado is not None:  # Verifica se um arquivo foi realmente enviado
        try:
            df = pd.read_csv(arquivo_enviado)  # Lê o arquivo CSV com o pandas
            df = df.dropna()  # Remove linhas com valores nulos
            return df
        except Exception as e:
            st.error(f"Erro ao ler o arquivo CSV: {e}")
            return None
    return None

def get_classificador(nome_classificador):
    """Cria os widgets de parâmetros na sidebar e retorna o modelo configurado."""
    # Esta função cria dinamicamente os "sliders" de parâmetros.
    # Ela só mostra os sliders relevantes para o modelo que o usuário escolheu.
    
    st.sidebar.subheader(f'Parâmetros do {nome_classificador}')
    params = {}  # Um dicionário para guardar os parâmetros escolhidos
    modelo = None

    if nome_classificador == "Árvore de Decisão":
        # st.sidebar.slider(label, min, max, default, step)
        profundidade = st.sidebar.slider("Profundidade da Árvore", 2, 20, 5, 1)
        params['max_depth'] = profundidade
        # 'random_state=42' garante que o modelo seja treinado da mesma forma toda vez (reprodutibilidade)
        modelo = DecisionTreeClassifier(max_depth=params['max_depth'], random_state=42)

    elif nome_classificador == "KNN (K-Nearest Neighbors)":
        n_vizinhos = st.sidebar.slider("Número de Vizinhos (K)", 1, 15, 5, 1)
        params['n_neighbors'] = n_vizinhos
        modelo = KNeighborsClassifier(n_neighbors=params['n_neighbors'])

    elif nome_classificador == "Regressão Logística":
        # 'C' é um parâmetro de regularização.
        C = st.sidebar.slider("Parâmetro C (Regularização)", 0.1, 10.0, 1.0, 0.1)
        params['C'] = C
        modelo = LogisticRegression(C=params['C'], random_state=42, max_iter=1000)

    elif nome_classificador == "SVM":
        C = st.sidebar.slider("Parâmetro C (Regularização)", 0.1, 10.0, 1.0, 0.1)
        params['C'] = C
        # Habilita 'probability=True' para que possamos usar 'predict_proba' mais tarde
        modelo = SVC(C=params['C'], random_state=42, probability=True)

    return modelo, params

# --- 4. Título e Introdução ---
# Comandos 'st.title' e 'st.write' desenham na tela principal da aplicação.
st.title("Análise e Predição de Pinguins Palmer ")
st.write("""
Aplicação web para análise visual e *Machine Learning* usando o dataset Palmer Penguins.
Explore os dados, visualize as relações e treine modelos para prever as espécies de pinguins.
""")

# --- 5. Sidebar (Barra Lateral) ---
# 'st.sidebar' coloca os elementos na barra lateral esquerda.

st.sidebar.header('1. Carregar Dados')
# 'st.sidebar.file_uploader' cria o widget de upload de arquivos.
arquivo_enviado = st.sidebar.file_uploader("Upload .csv (Opcional)", type=["csv"])

# Lógica principal de carregamento de dados
df = None
if arquivo_enviado is not None:
    # Se o usuário enviou um arquivo, vai carregar
    st.sidebar.success("CSV Carregado!")
    df = carregar_dados_upload(arquivo_enviado)
else:
    # Se não, carrega o dataset padrão
    st.sidebar.info("Usando dataset padrão (Palmer Penguins).")
    df = carregar_dados_padrao()

# --- 6. Corpo Principal da Aplicação ---

# O 'if df.empty:' é uma verificação de segurança.
# O resto da aplicação só vai rodar se os dados tiverem sido carregados com sucesso.
if df.empty:
    st.error("Nenhum dado para analisar. Faça upload de um CSV ou verifique o carregamento padrão.")
else:
    # 'st.dataframe' desenha uma tabela interativa
    st.header("Visão Geral dos Dados")
    st.dataframe(df.head())

    # --- 6.1. Análise de Dados e Visualização ---
    st.header("Análise Exploratória e Visualização")
    
    st.sidebar.header('2. Filtros de Análise')
    colunas_filtro_cat = ['island', 'sex']
    filtros = {}
    
    # Loop para criar os filtros dinamicamente
    for col in colunas_filtro_cat:
        if col in df.columns:  # Verifica se a coluna existe no dataframe
            opcoes = sorted(df[col].unique())
            # 'st.sidebar.multiselect' cria um menu de seleção múltipla
            selecionado = st.sidebar.multiselect(f'Filtrar por {col}:', opcoes, default=opcoes)
            filtros[col] = selecionado

    # Aplica os filtros selecionados no dataframe
    df_filtrado = df.copy()  # Copia o 'df' original para não alterá-lo
    for col, selecionado in filtros.items():
        # Esta é a linha de filtragem do Pandas
        df_filtrado = df_filtrado[df_filtrado[col].isin(selecionado)]

    st.write(f"Exibindo {df_filtrado.shape[0]} de {df.shape[0]} registros filtrados.")

    # 'st.columns(2)' cria um layout de duas colunas
    col1, col2 = st.columns(2)
    
    # 'with col1:' define o que vai entrar na primeira coluna
    with col1:
        st.subheader("Gráfico de Barras: Contagem de Espécies")
        # 'px.bar' cria um gráfico de barras interativo com Plotly
        fig_barra = px.bar(df_filtrado, x='species', color='species',
                           title="Contagem de Espécies (Filtrado)")
        # 'st.plotly_chart' exibe um gráfico Plotly no Streamlit
        st.plotly_chart(fig_barra, use_container_width=True)

    # 'with col2:' define o que vai entrar na segunda coluna
    with col2:
        st.subheader("Gráfico de Pizza: Distribuição por Ilha")
        df_ilhas = df_filtrado['island'].value_counts().reset_index()
        fig_pizza = px.pie(df_ilhas, names='island', values='count',
                           title="Distribuição por Ilha (Filtrado)")
        st.plotly_chart(fig_pizza, use_container_width=True)

    st.subheader("Gráfico Interativo: Relações entre Variáveis")
    # Esta seção cumpre o requisito de "flexibilidade", permitindo ao usuário
    # escolher quais colunas plotar.
    colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Tenta definir padrões (se existirem) para uma boa experiência inicial
    default_x = 'bill_length_mm' if 'bill_length_mm' in colunas_numericas else colunas_numericas[0]
    default_y = 'flipper_length_mm' if 'flipper_length_mm' in colunas_numericas else colunas_numericas[1]
    default_color = 'species' if 'species' in df.columns else None

    # Cria 3 colunas para os 3 menus 'selectbox'
    c1, c2, c3 = st.columns(3)
    eixo_x = c1.selectbox("Eixo X (Dispersão)", colunas_numericas, index=colunas_numericas.index(default_x))
    eixo_y = c2.selectbox("Eixo Y (Dispersão)", colunas_numericas, index=colunas_numericas.index(default_y))
    eixo_cor = c3.selectbox("Cor (Dispersão)", df.columns, index=list(df.columns).index(default_color) if default_color else 0)

    # Cria o gráfico de dispersão com base nas seleções do usuário
    fig_dispersao = px.scatter(
        df_filtrado, x=eixo_x, y=eixo_y, color=eixo_cor,
        hover_data=['island', 'sex'], title=f'{eixo_x} vs. {eixo_y}'
    )
    st.plotly_chart(fig_dispersao, use_container_width=True)

    # --- 6.2. Machine Learning (Treinamento) ---
    st.header("Machine Learning: Previsão de Espécies")
    
    st.sidebar.header('3. Configurar Modelo de ML')
    
    # Seleção da coluna Alvo (Y)
    coluna_alvo_default = 'species' if 'species' in df.columns else df.columns[0]
    coluna_alvo = st.sidebar.selectbox("Selecione a Coluna Alvo (Y)", df.columns, index=list(df.columns).index(coluna_alvo_default))
    
    # Seleção das colunas de Features (X)
    features_default = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    features_default_validas = [f for f in features_default if f in colunas_numericas]
    
    features_selecionadas = st.sidebar.multiselect(
        "Selecione as Features (X) (Numéricas)", 
        options=colunas_numericas, 
        default=features_default_validas
    )

    # Seleção do tipo de Classificador
    tipo_classificador = st.sidebar.selectbox(
        "Escolha o Classificador:",
        ("Árvore de Decisão", "KNN (K-Nearest Neighbors)", "Regressão Logística", "SVM")
    )
    
    # Chama nossa função para buscar o modelo e os sliders de parâmetros
    modelo, params = get_classificador(tipo_classificador)

    # Botão de Treinamento (Treinamento Dinâmico)
    # O código dentro deste 'if' SÓ é executado quando o usuário clica no botão.
    if st.sidebar.button("Treinar Modelo", type="primary"):
        if not features_selecionadas:
            st.error("Selecione pelo menos uma 'feature' para treinar o modelo.")
        else:
            st.subheader("Resultados do Treinamento")
            
            # 1. Preparar dados (X = features, Y = alvo)
            X = df[features_selecionadas]
            Y = df[coluna_alvo]
            
            # 2. Dividir os dados em conjuntos de treino e teste
            # test_size=0.3 significa 30% dos dados para teste, 70% para treino
            # stratify=Y garante que a proporção das classes (espécies) seja a mesma
            # nos conjuntos de treino e teste. Essencial para classificação.
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
            
            # 3. Treinar o modelo
            modelo.fit(X_train, Y_train)
            
            # 4. Avaliar o modelo
            Y_pred = modelo.predict(X_test)
            acuracia = accuracy_score(Y_test, Y_pred)
            st.write(f"**Classificador:** {tipo_classificador}")
            st.write(f"**Acurácia no Teste:** {acuracia:.2%}")

            # 5. Matriz de Confusão
            st.write("**Matriz de Confusão:**")
            cm = confusion_matrix(Y_test, Y_pred)
            labels = sorted(Y.unique())
            
            # Cria a figura do Matplotlib
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax_cm)
            ax_cm.set_xlabel('Previsto')
            ax_cm.set_ylabel('Verdadeiro')
            # 'st.pyplot' é como o Streamlit exibe gráficos do Matplotlib
            st.pyplot(fig_cm)
            
            # O Streamlit "reroda" o script do zero a cada interação.
            # Para "lembrar" o modelo treinado, nós o salvamos no 'st.session_state'
            # O 'session_state' é um dicionário que persiste entre os "reruns"
            st.session_state['modelo_treinado'] = modelo
            st.session_state['features_modelo'] = features_selecionadas
            st.success("Modelo treinado e pronto para predição!")

    # --- 6.3. Predição Dinâmica ---
    # Esta seção SÓ aparece se um modelo foi treinado e salvo no 'session_state'
    if 'modelo_treinado' in st.session_state:
        st.header("Faça uma Nova Predição")
        
        # Busca o modelo e as features salvas na sessão
        modelo_salvo = st.session_state['modelo_treinado']
        features_salvas = st.session_state['features_modelo']
        
        inputs_predicao = {}
        st.write("Ajuste os valores para prever a espécie:")
        
        # Cria dinamicamente N colunas para N sliders
        col_sliders = st.columns(len(features_salvas))
        
        # Loop para criar um slider para cada feature que o modelo usou
        for i, feature in enumerate(features_salvas):
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            default_val = float(df[feature].mean())
            
            # 'with col_sliders[i]:' coloca o slider na coluna 'i'
            with col_sliders[i]:
                inputs_predicao[feature] = st.slider(
                    label=feature, 
                    min_value=min_val, 
                    max_value=max_val, 
                    value=default_val,
                    step=0.1
                )
        
        # Botão para fazer a predição
        if st.button("Prever Espécie"):
            # 1. Monta um DataFrame de 1 linha com os dados dos sliders
            df_predicao = pd.DataFrame([inputs_predicao])
            
            # 2. Garante que a ordem das colunas é a MESMA que o modelo foi treinado
            df_predicao = df_predicao[features_salvas] 
            
            # 3. Faz a predição
            predicao_unica = modelo_salvo.predict(df_predicao)
            predicao_proba = modelo_salvo.predict_proba(df_predicao)
            
            # 4. Exibe o resultado
            st.subheader(f"Resultado da Predição: {predicao_unica[0]}")
            
            # 5. Exibe as probabilidades
            st.write("Probabilidades:")
            df_proba = pd.DataFrame(predicao_proba, columns=modelo_salvo.classes_)
            # Transpõe (gira) o dataframe para plotar com Plotly
            df_proba_transposed = df_proba.T.reset_index()
            df_proba_transposed.columns = ['Espécie', 'Probabilidade']
            
            fig_proba = px.bar(
                df_proba_transposed, 
                x='Espécie', 
                y='Probabilidade',
                color='Espécie'
            )
            st.plotly_chart(fig_proba, use_container_width=True)