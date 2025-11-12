import os
import pandas as pd
import seaborn as sns
import plotly.express as px
import joblib  # Usado para salvar e carregar o modelo treinado
from flask import Flask, render_template, request, redirect, url_for, session
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Cria a aplicação Flask
app = Flask(__name__)

# Chave secreta OBRIGATÓRIA para o Flask 'session' funcionar
# O 'session' é como um dicionário que "lembra" informações entre os reloads
app.secret_key = 'sua_chave_secreta_muito_segura_12345'

# Define caminhos
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'model'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# Garante que as pastas existam
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Define os caminhos dos arquivos que vamos usar
DATA_PATH = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')
MODEL_PATH = os.path.join(app.config['MODEL_FOLDER'], 'model.joblib')

def carregar_dados_padrao():
    """ Carrega os dados padrão (penguins) e salva em 'uploads/data.csv' """
    try:
        df = sns.load_dataset('penguins')
        df = df.dropna()
        df.to_csv(DATA_PATH, index=False)
        return True
    except Exception as e:
        print(f"Erro ao carregar dados padrão: {e}")
        return False

def gerar_graficos(df):
    """ Gera os gráficos com Plotly e retorna o HTML deles """
    try:
        # Gráfico 1: Contagem de Espécies (Barras)
        fig_barra = px.bar(df, x='species', color='species',
                           title="Contagem de Espécies")
        
        # Gráfico 2: Dispersão (Bico vs Nadadeira)
        fig_dispersao = px.scatter(df, x='bill_length_mm', y='flipper_length_mm',
                                   color='species',
                                   title='Comprimento do Bico vs. Nadadeira')
        
        # Converte os gráficos para HTML
        plot_html_barra = fig_barra.to_html(full_html=False, include_plotlyjs='cdn')
        plot_html_dispersao = fig_dispersao.to_html(full_html=False, include_plotlyjs='cdn')
        
        return plot_html_barra, plot_html_dispersao
    except Exception as e:
        print(f"Erro ao gerar gráficos: {e}")
        return None, None

# --- Rotas do Flask ---

@app.route('/')
def index():
    """ Rota principal - O 'painel' da nossa aplicação """
    
    # Tenta carregar dados e gráficos se um arquivo existir
    df = None
    plot_barra = None
    plot_dispersao = None
    
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        plot_barra, plot_dispersao = gerar_graficos(df)
        
    # 'render_template' carrega o 'index.html' e passa variáveis para ele
    return render_template(
        'index.html',
        # Passa o nome do arquivo (se existir) para o HTML
        nome_arquivo=session.get('nome_arquivo', None),
        # Passa os gráficos (se existirem) para o HTML
        plot_barra=plot_barra,
        plot_dispersao=plot_dispersao,
        # Passa o resultado do treinamento (se existir) para o HTML
        acuracia=session.get('acuracia', None),
        # Passa o resultado da predição (se existir) para o HTML
        predicao=session.get('predicao', None),
        # Verifica se o modelo já foi treinado
        modelo_pronto=os.path.exists(MODEL_PATH)
    )

@app.route('/upload', methods=['POST'])
def upload_file():
    """ Rota para lidar com o upload do CSV """
    
    # Pega o arquivo enviado no formulário
    file = request.files['file_csv']
    
    if file.filename == '':
        return redirect(url_for('index')) # Redireciona se nenhum arquivo foi selecionado

    if file:
        file.save(DATA_PATH)
        # Salva o nome do arquivo na 'session' para lembrarmos dele
        session['nome_arquivo'] = file.filename
        # Limpa resultados antigos
        session.pop('acuracia', None)
        session.pop('predicao', None)
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH) # Remove modelo antigo
            
    # Redireciona de volta para a página principal
    return redirect(url_for('index'))

@app.route('/default_data', methods=['POST'])
def use_default_data():
    """ Rota para usar os dados padrão 'penguins' """
    carregar_dados_padrao()
    session['nome_arquivo'] = 'penguins_padrao.csv'
    # Limpa resultados antigos
    session.pop('acuracia', None)
    session.pop('predicao', None)
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH) # Remove modelo antigo
        
    return redirect(url_for('index'))

@app.route('/train', methods=['POST'])
def train_model():
    """ Rota para treinar o modelo de ML """
    
    # Pega as escolhas do usuário no formulário de treinamento
    nome_classificador = request.form['classificador']
    
    # Parâmetros
    profundidade = int(request.form.get('profundidade_arvore', 5))
    n_vizinhos = int(request.form.get('n_vizinhos_knn', 5))
    
    # Carrega os dados que já foram 'uploadados'
    df = pd.read_csv(DATA_PATH)
    
    # Define features (X) e alvo (Y)
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    target = 'species'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Escolhe o modelo baseado na escolha do usuário
    if nome_classificador == "Árvore de Decisão":
        modelo = DecisionTreeClassifier(max_depth=profundidade, random_state=42)
    elif nome_classificador == "KNN":
        modelo = KNeighborsClassifier(n_neighbors=n_vizinhos)
    elif nome_classificador == "Regressão Logística":
        modelo = LogisticRegression(random_state=42, max_iter=1000)
    elif nome_classificador == "SVM":
        modelo = SVC(random_state=42, probability=True)
    else:
        # Padrão
        modelo = DecisionTreeClassifier(max_depth=5, random_state=42)

    # Treina o modelo
    modelo.fit(X_train, y_train)
    
    # Avalia
    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    
    # Salva o modelo treinado no disco!
    # 'joblib' é a forma padrão de salvar modelos do scikit-learn
    joblib.dump(modelo, MODEL_PATH)
    
    # Salva a acurácia na 'session' para mostrar ao usuário
    session['acuracia'] = f"{acuracia:.2%}"
    
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    """ Rota para fazer uma nova predição """
    
    # Verifica se o modelo já foi treinado e existe
    if not os.path.exists(MODEL_PATH):
        return redirect(url_for('index'))

    # Carrega o modelo salvo do disco
    modelo = joblib.load(MODEL_PATH)
    
    try:
        # Pega os dados do formulário de predição
        bill_length = float(request.form['bill_length'])
        bill_depth = float(request.form['bill_depth'])
        flipper_length = float(request.form['flipper_length'])
        body_mass = float(request.form['body_mass'])
        
        # Monta o DataFrame de 1 linha para o modelo
        features_predicao = pd.DataFrame(
            [[bill_length, bill_depth, flipper_length, body_mass]],
            columns=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
        )
        
        # Faz a predição
        predicao_unica = modelo.predict(features_predicao)
        
        # Salva o resultado na 'session'
        session['predicao'] = predicao_unica[0]
        
    except Exception as e:
        print(f"Erro na predição: {e}")
        session['predicao'] = "Erro nos dados"

    return redirect(url_for('index'))

# --- Ponto de Entrada ---
if __name__ == '__main__':
    # Roda a aplicação
    # debug=True faz o servidor reiniciar automaticamente quando você salva o arquivo
    app.run(debug=True)