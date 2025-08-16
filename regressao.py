import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Ler o CSV
dados = pd.read_csv("dados_empresas.csv")

# 2. Preparar os dados
X = dados[['Mes']]  # variável independente
y = dados['Emprestimos_Renegociados']  # variável dependente

# 3. Criar e treinar o modelo
modelo = LinearRegression()
modelo.fit(X, y)

# 4. Prever o próximo ponto
proximo_x = np.array([[dados['Mes'].max() + 1]])  # usa "Mes" em vez de "x"
proxima_previsao = modelo.predict(proximo_x)

print(f"Próximo valor previsto para Mes={proximo_x[0][0]} é Emprestimos_Renegociados={proxima_previsao[0]:.2f}")

# 5. Visualizar os dados e a reta de regressão
plt.scatter(X, y, color="blue", label="Dados reais")
plt.plot(X, modelo.predict(X), color="red", label="Reta de regressão")
plt.scatter(proximo_x, proxima_previsao, color="green", marker="x", s=100, label="Próxima previsão")

plt.xlabel("Mes")
plt.ylabel("Emprestimos_Renegociados")
plt.legend()
plt.show()
