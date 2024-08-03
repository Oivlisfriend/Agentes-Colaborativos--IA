import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import tkinter as tk
from tkinter import ttk
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning)

matriz_informacoes = [['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'] for _ in range(10)]
base = pd.read_csv('./IAdataset.csv')

def conversao(base):
    base.loc[base['Esquerda'] == 'b', 'Esquerda'] = '0'
    base.loc[base['Esquerda'] == 'l', 'Esquerda'] = '1'
    base.loc[base['Esquerda'] == 't', 'Esquerda'] = '2'
    base.loc[base['Esquerda'] == 'd', 'Esquerda'] = '3'
    base.loc[base['Direita'] == 'b', 'Direita'] = '0'
    base.loc[base['Direita'] == 'l', 'Direita'] = '1'
    base.loc[base['Direita'] == 't', 'Direita'] = '2'
    base.loc[base['Direita'] == 'd', 'Direita'] = '3'
    base.loc[base['Cima'] == 'b', 'Cima'] = '0'
    base.loc[base['Cima'] == 'l', 'Cima'] = '1'
    base.loc[base['Cima'] == 't', 'Cima'] = '2'
    base.loc[base['Cima'] == 'd', 'Cima'] = '3'
    base.loc[base['Baixo'] == 'b', 'Baixo'] = '0'
    base.loc[base['Baixo'] == 'l', 'Baixo'] = '1'
    base.loc[base['Baixo'] == 't', 'Baixo'] = '2'
    base.loc[base['Baixo'] == 'd', 'Baixo'] = '3'

    base = base.drop_duplicates(keep='last')
    X = base.drop('Target', axis=1)
    y = base['Target']
    return (X,y)

X,y = conversao(base);

class Ambiente:
    def __init__(self, tamanho, proporcao_lbt):
        self.tamanho = tamanho
        self.proporcao_lbt = proporcao_lbt
        self.matriz = self.inicializar_ambiente()
        self.adicionar_bandeira_aleatoria()  # Adiciona bandeira aleatória ao iniciar o ambiente

        self.root = tk.Tk()
        self.root.title("Ambiente")
        self.canvas = tk.Canvas(self.root, width=self.tamanho * 50, height=self.tamanho * 50)
        self.canvas.pack()
        btn_abordagem_A = ttk.Button(self.root, text="Abordagem A", command=lambda: simular_abordagem_A(X, y))
        btn_abordagem_A.pack()

        btn_abordagem_B = ttk.Button(self.root, text="Abordagem B", command=lambda: simular_abordagem_B(X, y))
        btn_abordagem_B.pack()

        btn_abordagem_C = ttk.Button(self.root, text="Abordagem C", command=lambda: simular_abordagem_C(X, y))
        btn_abordagem_C.pack()
        self.label_abordagem_A = ttk.Label(self.root, text="")
        self.label_abordagem_A.pack()
        self.label_abordagem_B = ttk.Label(self.root, text="")
        self.label_abordagem_B.pack()
        self.label_abordagem_C = ttk.Label(self.root, text="")
        self.label_abordagem_C.pack()

        # Definindo cores para os agentes
        self.cores_agentes = {"DecisionTreeClassifier": "orange", "KNeighborsClassifier": "lightblue", "GaussianNB": "lightgreen"}

    def inicializar_ambiente(self):
        matriz = [['L'] * self.tamanho for _ in range(self.tamanho)]
        bandeira_adicionada = False  # Variável para controlar se a bandeira já foi adicionada
        for i in range(self.tamanho):
            for j in range(self.tamanho):
                    r = random.random()
                    if r < self.proporcao_lbt['B']:
                        matriz[i][j] = 'B'
                    elif r < self.proporcao_lbt['B'] + self.proporcao_lbt['T']:
                        matriz[i][j] = 'T'
                    else:
                        matriz[i][j] = 'L'
        return matriz

    def adicionar_bandeira_aleatoria(self):
        i, j = random.randint(0, self.tamanho - 1), random.randint(0, self.tamanho - 1)
        self.matriz[i][j] = 'F'

    def imprimir_ambiente(self, agentes):
        self.canvas.delete("all")
        cores = {"L": "white", "B": "#f97171", "T": "#f9d171", "F": "#649bd7"}

        for i in range(self.tamanho):
            for j in range(self.tamanho):
                x0, y0 = j * 50, i * 50
                x1, y1 = x0 + 50, y0 + 50
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=cores[self.matriz[i][j]])

        for agente in agentes:
            if agente.vida>=0:
                x, y = agente.posicao
                x0, y0 = y * 50 + 10, x * 50 + 10
                x1, y1 = x0 + 30, y0 + 30
                cor_agente = self.cores_agentes[type(agente.modelo).__name__]
                self.canvas.create_oval(x0, y0, x1, y1, fill=cor_agente)

        self.root.update()

    def avaliar_resultados_abordagem_A(self):
        total_tesouros = sum([linha.count('T') for linha in self.matriz])
        tesouros_encontrados = sum([linha.count('T') for linha in matriz_informacoes])

        return tesouros_encontrados / total_tesouros if total_tesouros > 0 else 0

    def avaliar_resultados_abordagem_B(self, agentes):
        ambiente_total = self.tamanho * self.tamanho
        celulas_exploradas = 0
        total_agente = 0
        for linha in range(self.tamanho):
            for coluna in range(self.tamanho):
                if matriz_informacoes[linha][coluna] != 'D':
                    celulas_exploradas += 1
        for agente in agentes:
            if agente.vida >= 0:
                total_agente += 1
        total_explorado= celulas_exploradas / ambiente_total
        if (total_agente > 0) and (total_explorado == 1):
            return True
        return False

    def avaliar_resultados_abordagem_C(self):
        bandeira_encontrada = any(['F' in linha for linha in matriz_informacoes])
        return bandeira_encontrada

class Agente:
    def __init__(self, nome, modelo):
        self.nome = nome
        self.modelo = modelo
        self.posicao = (0, 0)
        self.movimentos = []
        self.estado = True
        self.vida = 0
        self.posicao_anterior = (0, 0)

    def treinar_modelo(self, dados_treino, rotulos_treino):
        self.modelo.fit(dados_treino, rotulos_treino)
        print(self.modelo.score(dados_treino, rotulos_treino))

    def fazer_predicao(self, dados):
        return self.modelo.predict(dados)

    def mover(self, ambiente, direcao):
        linha, coluna = self.posicao

        if direcao == 'cima':
            nova_posicao = (linha - 1, coluna)
        elif direcao == 'baixo':
            nova_posicao = (linha + 1, coluna)
        elif direcao == 'esquerda':
            nova_posicao = (linha, coluna - 1)
        elif direcao == 'direita':
            nova_posicao = (linha, coluna + 1)
        else:
            raise ValueError("Direção inválida!")

        if 0 <= nova_posicao[0] < ambiente.tamanho and 0 <= nova_posicao[1] < ambiente.tamanho:
            self.posicao_anterior=self.posicao
            self.posicao = nova_posicao
            self.movimentos.append((self.nome, self.posicao))
            print(f"{self.nome} moveu-se para {self.posicao}")
        else:
            print(f"{self.nome} tentou mover-se para uma posição inválida!")

    def interagir_ambiente(self, ambiente):
        linha, coluna = self.posicao

        if 0 <= linha < len(matriz_informacoes) and 0 <= coluna < len(matriz_informacoes[0]):
            celula_atual = ambiente.matriz[linha][coluna]

            if celula_atual == 'L':
                matriz_informacoes[linha][coluna] = 'L'
                print(f"{self.nome} está em uma célula livre.")
            elif celula_atual == 'B':
                self.vida-=1
                matriz_informacoes[linha][coluna] = 'B'
                self.estado=False
                print(f"{self.nome} encontrou uma bomba e foi destruído!")
            elif celula_atual == 'T':
                self.vida+=1
                matriz_informacoes[linha][coluna] = 'T'
                print(f"{self.nome} encontrou um tesouro e ficou mais forte!")
            elif celula_atual == 'F':
                matriz_informacoes[linha][coluna] = 'F'
                print(f"{self.nome} encontrou a bandeira!")
        else:
            print(f"{self.nome} está fora dos limites da matriz.")

    def tomar_acao(self, ambiente):
        linha, coluna = self.posicao

        # Coletar informações das posições adjacentes (cima, baixo, esquerda, direita)
        adjacentes = []
        for direcao in ['cima', 'baixo', 'direita', 'esquerda']:
            if direcao == 'cima':
                posicao_adjacente = (linha - 1, coluna)
            elif direcao == 'baixo':
                posicao_adjacente = (linha + 1, coluna)
            elif direcao == 'esquerda':
                posicao_adjacente = (linha, coluna - 1)
            elif direcao == 'direita':
                posicao_adjacente = (linha, coluna + 1)
            if 0 <= posicao_adjacente[0] < 10 and 0 <= posicao_adjacente[1] < 10:
                celula_adjacente = matriz_informacoes[posicao_adjacente[0]][posicao_adjacente[1]]
                if celula_adjacente == 'B':
                        adjacentes.append(0)
                elif celula_adjacente == 'L':
                    element = False

                    for agente in agentes:
                        if agente.posicao == posicao_adjacente:
                            element = True
                    if element or self.posicao_anterior == posicao_adjacente:
                        adjacentes.append(0)
                    else:
                        adjacentes.append(1)
                elif celula_adjacente == 'T':
                        if self.posicao_anterior == posicao_adjacente:
                            adjacentes.append(0)
                        else:
                            adjacentes.append(1)
                elif celula_adjacente == 'D':
                    adjacentes.append(3)
                else:
                    adjacentes.append(0)
            else:
                adjacentes.append(0)
        if all(adj == 0 for adj in adjacentes):
        # Encontrar uma adjacência válida e definir seu valor como 1
            for direcao, posicao_adjacente in zip(['cima', 'baixo', 'direita', 'esquerda'], [(linha - 1, coluna), (linha + 1, coluna), (linha, coluna + 1), (linha, coluna - 1)]):
                if 0 <= posicao_adjacente[0] < 10 and 0 <= posicao_adjacente[1] < 10:
                    index = ['cima', 'baixo', 'direita', 'esquerda'].index(direcao)
                    adjacentes[index] = 1
                    break
        # Fazer previsão com base nas informações coletadas
        acao_prevista = self.fazer_predicao([adjacentes])[0]
        # Executar ação prevista
        print(acao_prevista, adjacentes)
        self.mover(ambiente, acao_prevista)
        self.interagir_ambiente(ambiente)


def simular_exploracao(X,y):
    start_time = time.time()  # Registrar o tempo de início da simulação
    # Preparação dos dados para treinamento
    # Supondo que você tenha um conjunto de dados em formato CSV
    # Carregue o CSV e separe os recursos (X) e os rótulos (y)
    # Substitua esta parte do código com seu próprio processo de preparação de dados
    # Aqui está um exemplo genérico:


    # Treinamento dos modelos
    # Substitua os dados de treino e rótulos pelos seus próprios
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.33, random_state=1)
    for agente in agentes:
        agente.treinar_modelo(X_treino, y_treino)

    # Simulação da exploração
    num_passos_simulacao = 15  # Altere conforme necessário
    for passo in range(num_passos_simulacao):
        print(f"Passo {passo + 1}:\n")
        vida_agentes_maior_ou_igual_zero = sum(1 for agente in agentes if agente.vida < 0)
        if vida_agentes_maior_ou_igual_zero == len(agentes):
            ambiente.imprimir_ambiente(agentes)
            break
        for agente in agentes:
            if agente.vida >= 0:
                ambiente.imprimir_ambiente(agentes)
                agente.tomar_acao(ambiente)
                print(agente.vida)

    # Avaliação e comparação de resultados
    resultado_abordagem_A = ambiente.avaliar_resultados_abordagem_A()
    resultado_abordagem_B = ambiente.avaliar_resultados_abordagem_B(agentes)
    resultado_abordagem_C = ambiente.avaliar_resultados_abordagem_C()
    ambiente.label_abordagem_A.config(text=f"Resultados Abordagem A: {resultado_abordagem_A}")
    ambiente.label_abordagem_B.config(text=f"Resultados Abordagem B: {resultado_abordagem_B}")
    ambiente.label_abordagem_C.config(text=f"Resultados Abordagem C: {resultado_abordagem_C}")
    print(f"Resultados Abordagem A: {resultado_abordagem_A}")
    print(f"Resultados Abordagem B: {resultado_abordagem_B}")
    print(f"Resultados AboArdagem C: {resultado_abordagem_C}")

    end_time = time.time()  # Registrar o tempo de término da simulação
    execution_time = end_time - start_time  # Calcular o tempo total de execução
    print(f"Tempo de execução: {execution_time} segundos")

def simular_abordagem_A(X,y):
    start_time = time.time()  # Registrar o tempo de início da simulação

    # Treinamento dos modelos
    # Substitua os dados de treino e rótulos pelos seus próprios
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.33, random_state=1)
    for agente in agentes:
        agente.treinar_modelo(X_treino, y_treino)
    # Simulação da exploração
    num_passos_simulacao = 15  # Altere conforme necessário
    resultado_abordagem_A = ambiente.avaliar_resultados_abordagem_A()
    passo = 0;
    while resultado_abordagem_A * 100 < 50:
        passo += 1
        resultado_abordagem_A = ambiente.avaliar_resultados_abordagem_A()
        print(f"Passo {passo + 1}:\n")
        vida_agentes_maior_ou_igual_zero = sum(1 for agente in agentes if agente.vida < 0)
        if vida_agentes_maior_ou_igual_zero == len(agentes):
            ambiente.imprimir_ambiente(agentes)
            break
        for agente in agentes:
            if agente.vida >= 0:
                ambiente.imprimir_ambiente(agentes)
                agente.treinar_modelo(X_treino, y_treino)
                agente.tomar_acao(ambiente)
                print(agente.vida)

        resultado_abordagem_A = ambiente.avaliar_resultados_abordagem_A()

    # Avaliação e comparação de resultados
    resultado_abordagem_A = ambiente.avaliar_resultados_abordagem_A()
    ambiente.label_abordagem_A.config(text=f"Resultados Abordagem A: {resultado_abordagem_A*100}")
    print(f"Resultados Abordagem A: {resultado_abordagem_A}")

    end_time = time.time()  # Registrar o tempo de término da simulação
    execution_time = end_time - start_time  # Calcular o tempo total de execução

    print(f"Tempo de execução: {execution_time} segundos")
    for agente in agentes:
         agente.vida = 0
    for linha in range(10):
            for coluna in range(10):
                matriz_informacoes[linha][coluna]='D'


def simular_abordagem_B(X,y):
    start_time = time.time()  # Registrar o tempo de início da simulação

    # Treinamento dos modelos
    # Substitua os dados de treino e rótulos pelos seus próprios
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.33, random_state=1)
    for agente in agentes:
        agente.treinar_modelo(X_treino, y_treino)
    # Simulação da exploração
    passo = 0;
    resultado_abordagem_B = ambiente.avaliar_resultados_abordagem_B(agentes)
    while not resultado_abordagem_B:
        passo += 1
        print(f"Passo {passo + 1}:\n")
        vida_agentes_maior_ou_igual_zero = sum(1 for agente in agentes if agente.vida < 0)
        if vida_agentes_maior_ou_igual_zero == len(agentes):
            ambiente.imprimir_ambiente(agentes)
            break
        for agente in agentes:
            if agente.vida >= 0:
                ambiente.imprimir_ambiente(agentes)
                agente.tomar_acao(ambiente)
                agente.treinar_modelo(X_treino, y_treino)
                print(agente.vida)

        resultado_abordagem_B = ambiente.avaliar_resultados_abordagem_B(agentes)
    # Avaliação e comparação de resultados

    ambiente.label_abordagem_B.config(text=f"Resultados Abordagem B: {resultado_abordagem_B}")
    print(f"Resultados Abordagem B: {resultado_abordagem_B}")

    end_time = time.time()  # Registrar o tempo de término da simulação
    execution_time = end_time - start_time  # Calcular o tempo total de execução

    print(f"Tempo de execução: {execution_time} segundos")
    for agente in agentes:
             agente.vida = 0
    for linha in range(10):
            for coluna in range(10):
                matriz_informacoes[linha][coluna]='D'
def simular_abordagem_C(X,y):
    start_time = time.time()  # Registrar o tempo de início da simulação

    # Treinamento dos modelos
    # Substitua os dados de treino e rótulos pelos seus próprios
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.33, random_state=1)
    for agente in agentes:
        agente.treinar_modelo(X_treino, y_treino)
    # Simulação da exploração
    passo = 0;
    resultado_abordagem_C = ambiente.avaliar_resultados_abordagem_C()
    while not resultado_abordagem_C:
        passo += 1
        print(f"Passo {passo + 1}:\n")
        vida_agentes_maior_ou_igual_zero = sum(1 for agente in agentes if agente.vida < 0)
        if vida_agentes_maior_ou_igual_zero == len(agentes):
            ambiente.imprimir_ambiente(agentes)
            break
        for agente in agentes:
            if agente.vida >= 0:
                ambiente.imprimir_ambiente(agentes)
                agente.tomar_acao(ambiente)
                print(agente.vida)
                agente.treinar_modelo(X_treino, y_treino)
        resultado_abordagem_C = ambiente.avaliar_resultados_abordagem_C()
    # Avaliação e comparação de resultados



    ambiente.label_abordagem_C.config(text=f"Resultados Abordagem C: {resultado_abordagem_C}")
    print(f"Resultados AboArdagem C: {resultado_abordagem_C}")

    end_time = time.time()  # Registrar o tempo de término da simulação
    execution_time = end_time - start_time  # Calcular o tempo total de execução

    print(f"Tempo de execução: {execution_time} segundos")
    for agente in agentes:
        agente.vida = 0
    for linha in range(10):
            for coluna in range(10):
                matriz_informacoes[linha][coluna]='D'

# Exemplo de uso
tamanho_ambiente = 10
proporcao_lbt = {'L': 0.5, 'B': 0.3, 'T': 0.2, 'F': 0.1}
ambiente = Ambiente(tamanho_ambiente, proporcao_lbt)
# Não há agentes inicialmente
# Criação de agentes
agentes = []
num_agentes = 20  # Altere conforme necessário
for i in range(num_agentes):
    modelo_agente = DecisionTreeClassifier()
    agente = Agente(f'd{i + 1}', modelo_agente)
    agentes.append(agente)
#for i in range(num_agentes):
  # modelo = KNeighborsClassifier(n_neighbors=3)
  # agente = Agente(f'k{i + 1}', modelo)
  # agentes.append(agente)

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.33, random_state=1)
num_classes = len(set(y_treino))

# Ajustar os priors de acordo com o número de classes
priors = [1 / num_classes] * num_classes  # Distribuição uniforme de priors

# Criar o modelo GaussianNB com os priors ajustados

#for i in range(num_agentes):
 #  modelo = GaussianNB(priors=priors)
   #agente = Agente(f'n{i + 1}', modelo)
  # agentes.append(agente)

# Definição da interface gráfica

ambiente.root.mainloop()