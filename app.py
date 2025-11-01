# app.py - Aplicação Web com Streamlit

# --- 1. Importações ---
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Modelos
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings

# Ignorar warnings futuros (ex: de versões de bibliotecas)
warnings.filterwarnings('ignore')

# --- 2. Configuração da Página ---
st.set_page_config(
    layout="wide", # Usa a tela inteira
    page_title="Palmer Penguins App",
    page_icon="🐧"
)

# --- 3. Funções de Carregamento e ML ---

@st.cache_data # Cache para performance (não recarrega os dados a cada clique)
def carregar_dados_padrao():
    """Carrega o dataset 'penguins' do Seaborn e faz uma limpeza básica."""
    try:
        penguins = sns.load_dataset('penguins')
        penguins = penguins.dropna()
        return penguins
    except Exception as e:
        st.error(f"Erro ao carregar dados padrão: {e}")
        return pd.DataFrame()

def carregar_dados_upload(arquivo_enviado):
    """Lê um arquivo CSV enviado pelo usuário."""
    if arquivo_enviado is not None:
        try:
            df = pd.read_csv(arquivo_enviado)
            df = df.dropna()
            return df
        except Exception as e:
            st.error(f"Erro ao ler o arquivo CSV: {e}")
            return None
    return None

def get_classificador(nome_classificador):
    """Cria os widgets de parâmetros na sidebar e retorna o modelo configurado."""
    st.sidebar.subheader(f'Parâmetros do {nome_classificador}')
    params = {}
    modelo = None

    if nome_classificador == "Árvore de Decisão":
        profundidade = st.sidebar.slider("Profundidade da Árvore", 2, 20, 5, 1)
        params['max_depth'] = profundidade
        modelo = DecisionTreeClassifier(max_depth=params['max_depth'], random_state=42)

    elif nome_classificador == "KNN (K-Nearest Neighbors)":
        n_vizinhos = st.sidebar.slider("Número de Vizinhos (K)", 1, 15, 5, 1)
        params['n_neighbors'] = n_vizinhos
        modelo = KNeighborsClassifier(n_neighbors=params['n_neighbors'])

    elif nome_classificador == "Regressão Logística":
        C = st.sidebar.slider("Parâmetro C (Regularização)", 0.1, 10.0, 1.0, 0.1)
        params['C'] = C
        modelo = LogisticRegression(C=params['C'], random_state=42, max_iter=1000)

    elif nome_classificador == "SVM":
        C = st.sidebar.slider("Parâmetro C (Regularização)", 0.1, 10.0, 1.0, 0.1)
        params['C'] = C
        modelo = SVC(C=params['C'], random_state=42, probability=True) # Habilita probability

    return modelo, params

# --- 4. Título e Introdução ---
st.title("Análise e Predição de Pinguins Palmer 🐧")
st.write("""
Aplicação web para análise visual e *Machine Learning* usando o dataset Palmer Penguins.
Explore os dados, visualize as relações e treine modelos para prever as espécies de pinguins.
""")

# --- 5. Sidebar (Barra Lateral) ---
# A barra lateral é onde colocamos os controles principais.

st.sidebar.header('1. Carregar Dados')
arquivo_enviado = st.sidebar.file_uploader("Upload .csv (Opcional)", type=["csv"])

df = None
if arquivo_enviado is not None:
    st.sidebar.success("CSV Carregado!")
    df = carregar_dados_upload(arquivo_enviado)
else:
    st.sidebar.info("Usando dataset padrão (Palmer Penguins).")
    df = carregar_dados_padrao()

# --- 6. Corpo Principal da Aplicação ---

# Verifica se o dataframe foi carregado
if df.empty:
    st.error("Nenhum dado para analisar. Faça upload de um CSV ou verifique o carregamento padrão.")
else:
    # Mostra um preview dos dados
    st.header("Visão Geral dos Dados")
    st.dataframe(df.head())

    # --- 6.1. Análise de Dados e Visualização ---
    st.header("Análise Exploratória e Visualização")
    
    # Filtros na Sidebar (agora que o df está carregado)
    st.sidebar.header('2. Filtros de Análise')
    colunas_filtro_cat = ['island', 'sex']
    filtros = {}
    
    for col in colunas_filtro_cat:
        if col in df.columns:
            opcoes = sorted(df[col].unique())
            selecionado = st.sidebar.multiselect(f'Filtrar por {col}:', opcoes, default=opcoes)
            filtros[col] = selecionado

    # Aplicar filtros
    df_filtrado = df.copy()
    for col, selecionado in filtros.items():
        df_filtrado = df_filtrado[df_filtrado[col].isin(selecionado)]

    st.write(f"Exibindo {df_filtrado.shape[0]} de {df.shape[0]} registros filtrados.")

    # Gráficos (em colunas para organizar)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gráfico de Barras: Contagem de Espécies")
        fig_barra = px.bar(df_filtrado, x='species', color='species',
                           title="Contagem de Espécies (Filtrado)")
        st.plotly_chart(fig_barra, use_container_width=True)

    with col2:
        st.subheader("Gráfico de Pizza: Distribuição por Ilha")
        df_ilhas = df_filtrado['island'].value_counts().reset_index()
        fig_pizza = px.pie(df_ilhas, names='island', values='count',
                           title="Distribuição por Ilha (Filtrado)")
        st.plotly_chart(fig_pizza, use_container_width=True)

    st.subheader("Gráfico Interativo: Relações entre Variáveis")
    # Seleção de eixos para o gráfico de dispersão (Flexibilidade)
    colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Definir padrões se as colunas existirem
    default_x = 'bill_length_mm' if 'bill_length_mm' in colunas_numericas else colunas_numericas[0]
    default_y = 'flipper_length_mm' if 'flipper_length_mm' in colunas_numericas else colunas_numericas[1]
    default_color = 'species' if 'species' in df.columns else None

    c1, c2, c3 = st.columns(3)
    eixo_x = c1.selectbox("Eixo X (Dispersão)", colunas_numericas, index=colunas_numericas.index(default_x))
    eixo_y = c2.selectbox("Eixo Y (Dispersão)", colunas_numericas, index=colunas_numericas.index(default_y))
    eixo_cor = c3.selectbox("Cor (Dispersão)", df.columns, index=list(df.columns).index(default_color) if default_color else 0)

    fig_dispersao = px.scatter(
        df_filtrado, x=eixo_x, y=eixo_y, color=eixo_cor,
        hover_data=['island', 'sex'], title=f'{eixo_x} vs. {eixo_y}'
    )
    st.plotly_chart(fig_dispersao, use_container_width=True)

    # --- 6.2. Machine Learning ---
    st.header("Machine Learning: Previsão de Espécies")
    
    # Seleção de Modelo e Parâmetros na Sidebar
    st.sidebar.header('3. Configurar Modelo de ML')
    
    coluna_alvo_default = 'species' if 'species' in df.columns else df.columns[0]
    coluna_alvo = st.sidebar.selectbox("Selecione a Coluna Alvo (Y)", df.columns, index=list(df.columns).index(coluna_alvo_default))
    
    features_default = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    features_default_validas = [f for f in features_default if f in colunas_numericas]
    
    features_selecionadas = st.sidebar.multiselect(
        "Selecione as Features (X) (Numéricas)", 
        options=colunas_numericas, 
        default=features_default_validas
    )

    tipo_classificador = st.sidebar.selectbox(
        "Escolha o Classificador:",
        ("Árvore de Decisão", "KNN (K-Nearest Neighbors)", "Regressão Logística", "SVM")
    )
    
    modelo, params = get_classificador(tipo_classificador)

    # Botão de Treinamento (Treinamento Dinâmico)
    if st.sidebar.button("Treinar Modelo", type="primary"):
        if not features_selecionadas:
            st.error("Selecione pelo menos uma 'feature' para treinar o modelo.")
        else:
            st.subheader("Resultados do Treinamento")
            
            # 1. Preparar dados
            X = df[features_selecionadas]
            Y = df[coluna_alvo]
            
            # 2. Dividir
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
            
            # 3. Treinar
            modelo.fit(X_train, Y_train)
            
            # 4. Avaliar
            Y_pred = modelo.predict(X_test)
            acuracia = accuracy_score(Y_test, Y_pred)
            st.write(f"**Classificador:** {tipo_classificador}")
            st.write(f"**Acurácia no Teste:** {acuracia:.2%}")

            # 5. Matriz de Confusão
            st.write("**Matriz de Confusão:**")
            cm = confusion_matrix(Y_test, Y_pred)
            labels = sorted(Y.unique())
            
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax_cm)
            ax_cm.set_xlabel('Previsto')
            ax_cm.set_ylabel('Verdadeiro')
            st.pyplot(fig_cm)
            
            # Guardar modelo treinado no "estado" da sessão
            st.session_state['modelo_treinado'] = modelo
            st.session_state['features_modelo'] = features_selecionadas
            st.success("Modelo treinado e pronto para predição!")

    # --- 6.3. Predição Dinâmica ---
    # Só mostra esta seção se o modelo já foi treinado (está no session_state)
    if 'modelo_treinado' in st.session_state:
        st.header("Faça uma Nova Predição")
        
        modelo_salvo = st.session_state['modelo_treinado']
        features_salvas = st.session_state['features_modelo']
        
        # Criar sliders para cada feature usada no modelo
        inputs_predicao = {}
        st.write("Ajuste os valores para prever a espécie:")
        
        # Cria colunas para os sliders ficarem organizados
        col_sliders = st.columns(len(features_salvas))
        
        for i, feature in enumerate(features_salvas):
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            default_val = float(df[feature].mean())
            
            with col_sliders[i]:
                inputs_predicao[feature] = st.slider(
                    label=feature, 
                    min_value=min_val, 
                    max_value=max_val, 
                    value=default_val,
                    step=0.1 # Ajuste fino
                )
        
        # Botão para prever
        if st.button("Prever Espécie"):
            # Montar o DataFrame para predição
            df_predicao = pd.DataFrame([inputs_predicao])
            
            # Garantir a ordem das colunas
            df_predicao = df_predicao[features_salvas] 
            
            # Fazer a predição
            predicao_unica = modelo_salvo.predict(df_predicao)
            predicao_proba = modelo_salvo.predict_proba(df_predicao)
            
            st.subheader(f"Resultado da Predição: {predicao_unica[0]}")
            
            # Exibir Probabilidades
            st.write("Probabilidades:")
            df_proba = pd.DataFrame(predicao_proba, columns=modelo_salvo.classes_)
            df_proba_transposed = df_proba.T.reset_index()
            df_proba_transposed.columns = ['Espécie', 'Probabilidade']
            
            # Gráfico de barras das probabilidades
            fig_proba = px.bar(
                df_proba_transposed, 
                x='Espécie', 
                y='Probabilidade',
                color='Espécie'
            )
            st.plotly_chart(fig_proba, use_container_width=True)