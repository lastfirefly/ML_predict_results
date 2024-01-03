# Importar bibliotecas necessárias
import pandas as pd
#sklearn para machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from math import comb

# Carregar arquivo xlsx com os resultados anteriores para amostragem
arquivo_xlsx = 'archives/mega_sena_asloterias_ate_concurso_2669_sorteio.xlsx'
dados = pd.read_excel(arquivo_xlsx)

# Fazer limpeza dos dados - Remover colunas desnecessárias do arquivo
dados = dados.drop(['Concurso', 'Data'], axis=1, errors='ignore')

# Definir o intervalo de valores possíveis para os números presentes no jogo
valores_possiveis = range(1, 61)

# Dividir os dados em conjunto de treinamento e teste
dados_treino, dados_teste = train_test_split(dados, test_size=0.2, random_state=42)

# Criar e treinar um modelo de Random Forest Classifier para cada bola
modelos = {}
for bola in range(1, 7):
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(dados_treino.drop(f'bola {bola}', axis=1), dados_treino[f'bola {bola}'].astype(int))
    modelos[f'modelo_bola_{bola}'] = modelo

# Avaliar o desempenho do modelo no conjunto de teste
for bola in range(1, 7):
    modelo = modelos[f'modelo_bola_{bola}']
    previsoes = modelo.predict(dados_teste.drop(f'bola {bola}', axis=1))
    
    # Calcular o número de combinações possíveis para a bola atual
    num_combinacoes = comb(60, 6)
    
    # Imprimir o número de combinações possíveis
    print(f'Número de combinações possíveis para bola {bola}: {num_combinacoes}')
    
    relatorio_classificacao = classification_report(dados_teste[f'bola {bola}'].astype(int), previsoes, target_names=[str(i) for i in valores_possiveis])
    print(f'Resultados para bola {bola}:\n{relatorio_classificacao}')
    
    # Obter probabilidades previstas para o próximo jogo
    probabilidades = modelo.predict_proba(dados_teste.drop(f'bola {bola}', axis=1))
    
    # Calcular a média das probabilidades para cada número
    media_probabilidades = probabilidades.mean(axis=0)
    
    # Criar uma lista de tuplas (número, probabilidade média)
    lista_probabilidades = list(zip(valores_possiveis, media_probabilidades))
    
    # Classificar a lista em ordem decrescente de probabilidade
    lista_probabilidades = sorted(lista_probabilidades, key=lambda x: x[1], reverse=True)
    
    # Imprimir os 10 números mais prováveis para a próxima bola
    print(f'10 números mais prováveis para bola {bola}: {lista_probabilidades[:10]}')
    print('\n')
