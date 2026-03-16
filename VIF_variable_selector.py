import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def run_vif_cleaner(df, threshold=10.0):
    """
    Realiza un análisis de Factor de Inflación de la Varianza (VIF) iterativo.
    Elimina variables con alta multicolinealidad para asegurar la estabilidad del modelo.
    """
    # 1. Filtramos solo las columnas numéricas para el análisis estadístico
    # Ignoramos metadatos como 'class', 'year' o coordenadas brutas
    data = df.select_dtypes(include=['number']).drop(columns=['class', 'year'], errors='ignore')

    variables = data.columns.tolist()

    while True:
        # El cálculo de statsmodels requiere añadir una columna constante (intercepto)
        # Esto es vital para que el cálculo de R^2 sea correcto
        X = add_constant(data[variables])

        # Calculamos el VIF para cada una de las variables actuales
        vif_values = []
        for i in range(X.shape[1]):
            vif = variance_inflation_factor(X.values, i)
            vif_values.append(vif)

        # Creamos una serie para facilitar la búsqueda del máximo, quitando la constante
        vif_df = pd.Series(vif_values, index=X.columns).drop('const')

        # Identificamos el valor más alto
        max_vif = vif_df.max()

        if max_vif > threshold:
            var_to_drop = vif_df.idxmax()
            print(f"He eliminado {var_to_drop} porque su valor era {max_vif:.2f}")
            variables.remove(var_to_drop)
        else:
            print("El análisis ha terminado porque todas las variables supervivientes están bajo el umbral.")
            break

    return variables, vif_df.sort_values(ascending=False)