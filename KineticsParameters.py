# Development Branch. The work is done here, and then merged into the master branch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from re import search
import os

# Constantes globales
R = 8.3144              # Constante de los gases en J/mol·K

# ====================================================
# 1. CARGA Y PROCESAMIENTO DE DATOS
# ====================================================
def cargar_datos(ruta_archivo):
    """
    Se cargan los datos del archivo .csv a un dataframe (df) y se sustituyen las ',' que hay en los valores de las columnas por '.'
    Ademas se convierten dichos valores a float y se devuelve el df con las modificaciones."""
    df = pd.read_csv(ruta_archivo, sep=';')
    for col in ['Temperature T(c)', 'Time t (min)', 'Weight (mg)', 'Weight (%)', 'Heat Flow Q (mW)', 'Heat Flow (Normalized) Q (W/g)']:
        df[col] = df[col].str.replace(',', '.').astype(float)
    return df

def procesar_datos(df):
    # Convertir las columnas de Pandas a arrays de Numpy
    time = np.array(df['Time t (min)'])
    temperature = np.array(df['Temperature T(c)'])
    weight_percent = np.array(df['Weight (%)'])
    weight_mg = np.array(df['Weight (mg)'])
    heat_flow_q = np.array(df['Heat Flow Q (mW)'])
    #heat_flow_normalized = np.array(df['Heat Flow (Normalized) Q (W/g)'])

    # Conversión de temperatura a Kelvin
    temperature_k = temperature + 273

    # Calcular fracción de conversión (alpha) - Nos dice la cantidad de material que se ha convertido
    weight_0, weight_f = weight_mg[0], weight_mg[-1]
    alpha = np.clip((weight_0 - weight_mg) / (weight_0 - weight_f), 0, 1)

    # Filtrar valores de alpha válidos
    valid_alpha_mask = (alpha > 0) & (alpha < 1)

    # Devuelve un diccionario que alberga los datos filtrados
    return {
        'time': time,
        'temperature_k': temperature_k[valid_alpha_mask],
        'alpha': alpha[valid_alpha_mask],
        'weight_mg': weight_mg,
        'weight_percent': weight_percent,
        'heat_flow_q': heat_flow_q,
        'temperature': temperature
    }

# ====================================================
# 2. CÁLCULO DE DTG Y SUAVIZADO
# ====================================================
def calcular_dtg(temperature, weight_mg):   # Obtención de la curva DTG usando metodo numerico de diferencias finitas
    dtg = np.zeros_like(weight_mg)  # Crear un array vacío para la DTG del mismo tamaño que el dado, en este caso; weight_mg
    for i in range(1, len(temperature) - 1):
        # Evitar división por cero
        delta_temp = temperature[i] - temperature[i-1]
        dtg[i] = (weight_mg[i] - weight_mg[i-1]) / delta_temp if delta_temp != 0 else 0

    # En los extremos (primer y último punto), se usa la diferencia hacia adelante y hacia atrás
    dtg[0] = (weight_mg[1] - weight_mg[0]) / (temperature[1] - temperature[0]) if (temperature[1] - temperature[0]) != 0 else 0
    dtg[-1] = (weight_mg[-1] - weight_mg[-2]) / (temperature[-1] - temperature[-2]) if (temperature[-1] - temperature[-2]) != 0 else 0

    # Reemplazar valores infinitos o NaN en la curva DTG para evitar errores en la gráfica
    return np.nan_to_num(dtg, nan=0.0, posinf=0.0, neginf=0.0)

#Funcion para eliminar el ruido mediante la técnica del promedio movil (Moving Average)
def suavizar_dtg(dtg, window_size):
    # Crear un array de promedios móviles, usando una ventana deslizante
    return np.convolve(dtg, np.ones(window_size) / window_size, mode='same')

# ====================================================
# 3. DIVISIÓN EN SUBGRUPOS POR TEMPERATURA
# ====================================================
# Función para encontrar el número más cercano en el array de Temperature, ya que la temperatura que introduce el usuario existe en el array de temperaturas
def encontrar_mas_cercano(arr, num):
    idx = (np.abs(arr - num)).argmin()
    return arr[idx],idx


def dividir_en_subgrupos(temperature, temperature_k, alpha, weight_mg, time, heat_flow_q, temp_subgrupos):
    # Encontrar los índices de las temperaturas seleccionadas o las más cercanas
    indices = []
    for temp in temp_subgrupos:
        temp_cercana, idx = encontrar_mas_cercano(temperature, temp)
        # print( f"La temperatura más cercana a {temp:.2f} en el vector de Temperatura es {temp_cercana:.2f} en el índice {idx}")
        indices.append(idx)  # Añade el indice idx (el indice de la temp mas cercana) a lista de indices

    # Ordenar los índices por si no están en orden y eliminar duplicados
    indices = sorted(set(indices))

    # Dividir los arrays en subgrupos según esos índices. Retorna un diccionario que contiene los valores de los subgrupos de
    # cada una de las variables. subgrupos['temp'] o subrupos.get('temp') = [[valores subgrupo 1], [valores subgrupo 2]...]
    subgrupos = {
        'temp': np.split(temperature, indices),
        'temp_K': np.split(temperature_k, indices),
        'alpha': np.split(alpha, indices),
        'weight': np.split(weight_mg, indices),
        'time': np.split(time, indices),
        'heat_flow_q': np.split(heat_flow_q, indices)
    }
    return subgrupos, indices

# ====================================================
# 4. MODELOS CINÉTICOS
# ====================================================
# Funciones g(alpha) para los distintos modelos cinéticos
# Chemical reaction model
def cr0(alpha): return alpha                                            # Zero order (CR0)
def cr1(alpha): return -np.log(1 - alpha)                               # First order (CR1)
def cr2(alpha): return 2 * ((1 - alpha) ** (-1.5) - 1)                  # Second order (CR2)
def cr3(alpha): return 0.5 * ((1 - alpha) ** (-2) - 1)                  # Third order (CR3)

# Diffusion model
def dm0(alpha): return alpha**2                                         # Parabolic law (DM0)
def dm1(alpha): return alpha + (1 - alpha) * np.log(1 - alpha)          # Valensi (2D diffusion) (DM1)
def dm2(alpha): return (1 - (2*alpha/3)) - (1 - alpha)**(2/3)           # Ginstling - Broushtein (DM2)

# Nucleation and growth model
def ng1_5(alpha): return (-np.log(1 - alpha))**(2/3)                    # Avrami - Erofeev (n = 1.5) (NG1.5)
def ng2(alpha): return (-np.log(1 - alpha))**(1/2)                      # Avrami - Erofeev (n = 2) (NG2)
def ng3(alpha): return (-np.log(1 - alpha))**(1/3)                      # Avrami - Erofeev (n = 3) (NG3)

modelos = {
    "CR0": cr0, "CR1": cr1, "CR2": cr2, "CR3": cr3,
    "DM0": dm0, "DM1": dm1, "DM2": dm2,
    "NG1.5": ng1_5, "NG2": ng2, "NG3": ng3
}

# ====================================================
# 5. AJUSTE Y PARÁMETROS
# ====================================================
def ajustar_modelo(temperature, alpha, modelo_func, beta):
    t_inv = 1 / temperature
    g_alpha = modelo_func(alpha)
    # Variables para la regresión, LinearRegression espera que los datos de entrada x (variables independientes) tengan
    # dos dimensiones, donde cada fila representa un punto de datos y cada columna una variable. Por lo tanto, reshape
    # transforma el vector 1D en un matriz con una sola columna.
    x = t_inv.reshape(-1, 1)
    y = np.log(g_alpha / temperature ** 2)

    # Ajuste lineal con sklearn y Parámetros del ajuste
    reg = LinearRegression()
    reg.fit(x,y)

    pendiente, intercepto = reg.coef_[0], reg.intercept_
    r2 = reg.score(x, y)

    # Calcular energía de activación (Ea) y factor pre-exponencial (A)
    Ea = -pendiente * R                         # Energía de activación en J/mol
    A = (beta * Ea / R) * np.exp(intercepto)    # Factor pre-exponencial min^-1

    ajuste_reg = pendiente * (1 / temperature) + intercepto

    return Ea, A, r2, pendiente, intercepto, x, y, ajuste_reg


def aplicar_modelos(subgrupos, beta):
    # Crear un diccionario para almacenar los resultados de los modelos ajustados en cada subgrupo
    resultados = {}
    for i, (temp_K_sub, alpha_sub) in enumerate(zip(subgrupos['temp_K'], subgrupos['alpha'])): # Iterar sobre cada subgrupo y ajustar todos los modelos
        subgrupo_key = f"{i+1}"    # Clave para el subgrupo en el diccionario
        # Crear un sub-diccionario para cada subgrupo
        if subgrupo_key not in resultados:
            resultados [subgrupo_key] = {}

        for nombre, modelo_func in modelos.items():
            Ea, A, r2, pendiente, ordenada,x,y,ajuste_reg = ajustar_modelo(temp_K_sub, alpha_sub, modelo_func, beta)
            # Guardar los resultados del modelo dentro del subgrupo
            resultados[subgrupo_key][nombre]={
                "Ea (J/mol)":Ea,
                "A (1/min)": A,
                "R^2": r2,
                "reg_x": x,
                "reg_y": y,
                "ajuste_reg": ajuste_reg,
                "a": pendiente,
                "b": ordenada,
            }
    return resultados

# ====================================================
# 6. GUARDAR RESULTADOS, GRAFICAR y GUARDAR GRAFICOS
# ====================================================
def guardar_resultados(resultados, ruta_archivo):

    # Convertir el diccionario de resultados a un Dataframe
    resultados_list =[] # Creacion lista vacia

    for subgrupo,modelos in resultados.items():
        for modelo, valores in modelos.items():
            # Creacion de una fila solo con las columnas necesarias (omitiendo en este caso reg_x, reg_y y ajuste_reg
            fila = {
                "Subgrupo": subgrupo,
                "Modelo": modelo,
                "Ea (J/mol)": valores["Ea (J/mol)"],
                "A (1/min)": valores["A (1/min)"],
                "R2": valores["R^2"],
                "Pendiente":valores ["a"],
                "Ordenada": valores ["b"]
            }
            resultados_list.append(fila)

    # Convertir la lista de diccionarios en un Dataframe
    df_resultados = pd.DataFrame(resultados_list)

    # Obtener el nombre del archivo sin la extensión (ejemplo: 'celulosa_15_aire')
    nombre_archivo = os.path.splitext(os.path.basename(ruta_archivo))[0]

    # Determinar la subcarpeta (VelocidadCalentamiento5, VelocidadCalentamiento15, VelocidadCalentamiento30)
    subcarpeta = os.path.dirname(ruta_archivo).split(os.sep)[-1]

    # Crear la ruta del directorio donde se guardará el archivo de Excel
    ruta_directorio_salida = os.path.join('Resultados', 'ParametrosCineticos', subcarpeta)

    # Crear el directorio si no existe
    os.makedirs(ruta_directorio_salida, exist_ok=True)

    # Crear el nombre del archivo Excel (por ejemplo: 'celulosa_15_aire.xlsx')
    nombre_archivo_excel = f'{nombre_archivo}.xlsx'

    # Ruta completa del archivo Excel
    ruta_salida = os.path.join(ruta_directorio_salida, nombre_archivo_excel)

    # Guardar el DataFrame en un archivo Excel
    df_resultados.to_excel(ruta_salida, index=False)
    print(f"Resultados guardados en '{ruta_salida}'")

# Función para generar gráficos con múltiples ejes Y y particiones
def generar_grafico(df, eje_x, columnas_y, ruta_archivo, particiones=None):
    if not columnas_y:
        print("No se seleccionaron columnas para el eje Y. Por favor, especifica al menos una columna.")
        return

    # Comprobar si las columnas seleccionadas existen en el DataFrame
    columnas_validas = [col for col in columnas_y if col in df.columns]
    if not columnas_validas:
        print(f"Ninguna de las columnas seleccionadas existe en el DataFrame. Columnas disponibles: {list(df.columns)}")
        return

    nombre_archivo = os.path.splitext(os.path.basename(ruta_archivo))[0]
    fig, ax1 = plt.subplots(figsize=(12,8))

    # Configurar el eje X
    eje_x_data = df[eje_x]
    ax1.set_xlabel(eje_x)

    # Crear un diccionario de colores para asignar a cada serie
    colores = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow']

    # Inicializar el primer eje
    ax = ax1
    ejes_y = [ax1]

    # Graficar cada variable del eje Y según el orden de entrada
    for i, col in enumerate(columnas_validas):
        if i == 0:  # Primer eje Y primario
            ax.plot(eje_x_data, df[col], color=colores[i % len(colores)], label='_nolegend_')
            ax.set_ylabel(col, color=colores[i % len(colores)])
            ax.tick_params(axis='y', labelcolor=colores[i % len(colores)])
        else:
            # Crear un nuevo eje Y adicional
            ax = ax1.twinx()
            ax.spines["right"].set_position(("axes", 1 + 0.1 * (i - 1)))  # Mover el eje a la derecha en incrementos
            ax.plot(eje_x_data, df[col], color=colores[i % len(colores)], label=col)
            ax.set_ylabel(col, color=colores[i % len(colores)])
            ax.tick_params(axis='y', labelcolor=colores[i % len(colores)])
            ejes_y.append(ax)

    # Añadir líneas discontinuas en las posiciones donde se hacen las particiones
    if particiones is not None:
        for idx in particiones:
            valor_x = eje_x_data[idx]  # Obtener el valor del eje X correspondiente a la partición
            ax1.axvline(x=valor_x, color='gray', linestyle='--', label=f'Partición en {valor_x:.2f}')
            ax1.legend(loc='upper right')

    # Ajustar la figura para evitar solapamientos
    fig.suptitle(f"Gráfico de {nombre_archivo}")
    fig.tight_layout()
    #plt.show()

    return fig

def guardar_graficas (figura, ruta_archivo):
    # Determinar la subcarpeta (VelocidadCalentamiento5, VelocidadCalentamiento15, VelocidadCalentamiento30)
    carpeta = os.path.dirname(ruta_archivo).split(os.sep)[-2]
    subcarpeta = os.path.dirname(ruta_archivo).split(os.sep)[-1]

    nombre_archivo = os.path.splitext(os.path.basename(ruta_archivo))[0]

    # Crear el directorio correspondiente
    if carpeta == "DatosBiomasa":
        directorio_graficas = f'Resultados/GraficasBiomasa/{subcarpeta}'
    else:
        directorio_graficas = f'Resultados/Graficas/{subcarpeta}'

    os.makedirs(directorio_graficas, exist_ok=True)

    # Guardar la figura en el directorio adecuado
    ruta_grafico = os.path.join(directorio_graficas, f'{nombre_archivo}.png')
    figura.savefig(ruta_grafico, dpi=300)  # Guarda la gráfica
    plt.close(figura)  # Cierra la figura para liberar memoria

    print(f"\nGráfico guardado en: {ruta_grafico}")

# ====================================================
# 7. ENERGÍA ASOCIADA AL PROCESO
# ====================================================
def calcular_energia(heat_flow_q, time):
    heat_flow_w = heat_flow_q * 0.001
    time_s = time * 60
    energia = np.trapezoid(heat_flow_w, time_s)
    print(f"La energia obtenida de la integral temporal de la curva DSC es: {energia: .4f} J")

# ====================================================
# 8. REPRESENTACION DE LAS REGRESIONES
# ====================================================
# ====================================================
# Función para la representación gráfica de la regresiones
# ====================================================
def representar_regresion(resultados, ruta_archivo):

    nombre_componente= os.path.splitext(os.path.basename(ruta_archivo))[0]
    #print(f"Nombre Componente: {nombre_componente}")

    for subgrupo_key, subgrupo_val in resultados.items():
        # Crear la figura con una cuadrícula de 2 filas x 5 columnas - Albergara todos los modelos de un subgrupo
        fig, axs = plt.subplots(2, 5, figsize=(20, 10))

        for idx, (nombre, datos) in enumerate(subgrupo_val.items()):
            #rint(f"Subgrupo: {subgrupo_key} - Nombre: {nombre}")
            Ea = datos['Ea (J/mol)']
            A = datos['A (1/min)']
            R2 = datos['R^2']
            #print(f" Ea:{Ea}\n A:{A}\n R2:{R2}")

            reg_x = datos["reg_x"]
            reg_y = datos["reg_y"]
            ajuste_reg = datos["ajuste_reg"]

            row = idx // 5
            col = idx % 5

            # Determinar la posición de la gráfica en la fila inferior
            axs[row, col].plot(reg_x, reg_y, linestyle='--', label=f'ln(g(alpha) / T^2) vs 1/T')
            axs[row, col].plot(reg_x, ajuste_reg, linestyle='-',
                               label=f'Ajuste lineal con modelo {nombre}:\nEa = {Ea:.2f} kJ/mol\nA = {A:.2e} s⁻¹\nr² = {R2:.4f}')

            # Configurar cada gráfica de la fila inferior
            axs[row, col].set_xlabel('1/T')
            axs[row, col].set_ylabel('ln(g(alpha) / T^2)')
            axs[row, col].set_title(f'{nombre}')
            axs[row, col].grid(True)
            axs[row, col].legend(loc="best", fontsize="small")

            # Rotar las etiquetas del eje x para evitar solapamiento
            axs[row, col].tick_params(axis='x', rotation=45)

        # Devuelve las temperaturas iniciales y finales de cada subgrupo
        temp_inicial = int((1 / reg_x[0]) - 273)
        temp_final = int((1 / reg_x[-1]) - 273)

        # Ajustar diseño de los subplots
        fig.tight_layout()

        # Añadir margen y el título general
        plt.subplots_adjust(top=0.88)  # Margen superior adicional para el título
        fig.suptitle(f"Ajustes lineales para el subgrupo {subgrupo_key} ({temp_inicial}ºC - {temp_final}ºC) del componente {nombre_componente}",
                     fontsize=16, y=0.95)  # Posición ligeramente más abajo
        # plt.show()

        # Añadir esto en la funcion de guardar grafica igual.
        # Determinar la subcarpeta (VelocidadCalentamiento5, VelocidadCalentamiento15, VelocidadCalentamiento30)
        subcarpeta = os.path.dirname(ruta_archivo).split(os.sep)[-1]

        nombre_archivo = nombre_componente

        # Crear el directorio correspondiente
        directorio_regresiones = f'Resultados/Regresiones/{subcarpeta}'
        os.makedirs(directorio_regresiones, exist_ok=True)

        # Guardar la figura en el directorio adecuado
        ruta_regresiones = os.path.join(directorio_regresiones, f'{nombre_archivo}_subgrupo{subgrupo_key}.png')
        plt.savefig(ruta_regresiones, dpi=300)  # Guarda la gráfica
        plt.close(fig)  # Cierra la figura para liberar memoria

        print(f"Grafico regresiones guardado en: {ruta_regresiones}")


def obtencion_diccionario (ruta_resultados):

    df1 = pd.read_excel (ruta_resultados)
    # Asegúrate de que el DataFrame se ha cargado correctamente
    print(df1.head())

    # Convertir a un diccionario anidado por Subgrupo y Modelo
    diccionario_anidado = {}

    """ Iterar sobre cada fila del Datafram como una serie de Pandas, Devuelve un par (indice, fila) donde "_" es una convencion para indicar que la variable de indice no se va a utilizar
    , fila es un objeto de tipo Series que representa los datos de la fila actual. Recordar que la estructura de un dataframe de Pandas es del estilo de:
           Subgrupo Modelo     Ea (J/mol)  ...        R2     Pendiente   Ordenada
    0         1    CR0   53687.722000  ...  0.800811  -6457.197393   3.682499
    1         1    CR1   54338.567886  ...  0.805614  -6535.476749   3.934011
    2         1    CR2   55331.317344  ...  0.812714  -6654.877964   5.416142
    3         1    CR3   55666.626976  ...  0.815052  -6695.206747   4.447035
    4         1    DM0  112956.033739  ...  0.817032 -13585.590510  21.000513

    """
    for _, fila in df1.iterrows():
        subgrupo = fila['Subgrupo']
        modelo = fila['Modelo']

        # Crear una entrada para el subgrupo si no existe
        if subgrupo not in diccionario_anidado:
            diccionario_anidado[subgrupo] = {}

        # Agregar el modelo y sus parámetros
        diccionario_anidado[subgrupo][modelo] = {
            "Ea (J/mol)": fila['Ea (J/mol)'],
            "R2": fila['R2'],
            "a": fila['Pendiente'],
            "b": fila['Ordenada']
        }
    return diccionario_anidado

def calculo_porcentajes_modelo1 (parametros_biomasa, parametros_componentes):
    A = np.array([[parametros_componentes["celulosa_15_aire"]["1"]["CR0"]["a"], parametros_componentes["lignina_15_aire"]["1"]["CR0"]["a"], parametros_componentes["xilano_15_aire"]["1"]["CR0"]["a"]],
                  [parametros_componentes["celulosa_15_aire"]["1"]["CR0"]["b"], parametros_componentes["lignina_15_aire"]["1"]["CR0"]["b"], parametros_componentes["xilano_15_aire"]["1"]["CR0"]["b"]],
                  [1, 1, 1]])

    B = np.array([parametros_biomasa["hoja_enebro_15_aire"]["1"]["CR3"]["a"], parametros_biomasa["hoja_enebro_15_aire"]["1"]["CR3"]["b"], 1])
    solucion = np.linalg.solve(A, B)

    print("La solución es para el subgrupo 1 de la Hoja Enebro 15 aire es:")
    print("%Celulosa =", solucion[0]*100)
    print("%Lignina =", solucion[1]*100)
    print("%Xilano =", solucion[2]*100)

    # ----------------------------------
    A = np.array([[parametros_componentes["celulosa_15_aire"]["2"]["NG3"]["a"], parametros_componentes["lignina_15_aire"]["2"]["NG3"]["a"], parametros_componentes["xilano_15_aire"]["3"]["NG3"]["a"]],
                  [parametros_componentes["celulosa_15_aire"]["2"]["NG3"]["b"], parametros_componentes["lignina_15_aire"]["2"]["NG3"]["b"], parametros_componentes["xilano_15_aire"]["3"]["NG3"]["b"]],
                  [1, 1, 1]])

    B = np.array([parametros_biomasa["hoja_enebro_15_aire"]["2"]["CR0"]["a"], parametros_biomasa["hoja_enebro_15_aire"]["2"]["CR0"]["b"], 1])
    solucion = np.linalg.solve(A, B)

    print("La solución es para el subgrupo 2 de la Hoja Enebro 15 aire es:")
    print("%Celulosa =", solucion[0] * 100)
    print("%Lignina =", solucion[1] * 100)
    print("%Xilano =", solucion[2] * 100)

    # ----------------------------------
    A = np.array([[parametros_componentes["celulosa_15_aire"]["3"]["CR1"]["a"], parametros_componentes["lignina_15_aire"]["3"]["DM2"]["a"], parametros_componentes["xilano_15_aire"]["1"]["CR0"]["a"]],
                  [parametros_componentes["celulosa_15_aire"]["3"]["CR1"]["b"], parametros_componentes["lignina_15_aire"]["3"]["DM2"]["b"], parametros_componentes["xilano_15_aire"]["5"]["CR0"]["b"]],
                  [1, 1, 1]])

    B = np.array([parametros_biomasa["hoja_enebro_15_aire"]["1"]["CR3"]["a"], parametros_biomasa["hoja_enebro_15_aire"]["1"]["CR3"]["b"], 1])
    solucion = np.linalg.solve(A, B)

    print("La solución es para el subgrupo 3 de la Hoja Enebro 15 aire es:")
    print("%Celulosa =", solucion[0] * 100)
    print("%Lignina =", solucion[1] * 100)
    print("%Xilano =", solucion[2] * 100)

# ============================================================================
# FLUJO PRINCIPAL PARA EL ANALISIS CINETICO DE LOS COMPONENTES INDIVIDUALMENTE
# ============================================================================
def main_componentes(ruta_archivo):
    # Seleccion temperaturas para división en subgrupos (distintas fases de los componentes)
    temp_seleccionadas = {
        'celulosa_5_aire': [290, 335, 530],
        'celulosa_5_n2': [295, 340, 500],
        'lignina_5_aire': [200, 420],
        'lignina_5_n2': [190, 600],
        'xilano_5_aire': [240, 265, 450],
        'xilano_5_n2': [245, 280, 600],
        'celulosa_15_aire': [100,300,350,575],   # Añadidos para realizar mejores ajustes en cada fase, de cara a obtener el porcentaje masico de cada componente en la biomasa
        #'celulosa_15_aire': [310, 350, 575],
        'celulosa_15_n2': [315, 360, 600],
        'lignina_15_aire': [100,225,630],
        #'lignina_15_aire': [225, 630],
        'lignina_15_n2': [200, 580],
        'xilano_15_aire': [50,115,225,285,520],
        #'xilano_15_aire': [255, 285, 520],
        'xilano_15_n2': [270, 290, 600],
        'celulosa_30_aire': [320, 375, 600],
        'celulosa_30_n2': [330, 380, 600],
        'lignina_30_aire': [225, 850],
        'lignina_30_n2': [200, 440],
        'xilano_30_aire': [265, 300, 645],
        'xilano_30_n2': [280, 310, 600],
    }

    nombre_componente = os.path.splitext(os.path.basename(ruta_archivo))[0]    # celulosa_5_aire, por ejemplo.
    temp_subgrupos = temp_seleccionadas[nombre_componente]

    df = cargar_datos(ruta_archivo)
    datos = procesar_datos(df)

    dtg = calcular_dtg(datos['temperature'], datos['weight_mg'])
    dtg_suavizado = suavizar_dtg(dtg, 400)
    df['DTG_suavizado'] = dtg_suavizado

    subgrupos, indices = dividir_en_subgrupos(datos['temperature'], datos['temperature_k'], datos['alpha'], datos['weight_mg'], datos['time'], datos['heat_flow_q'], temp_subgrupos)

    figura = generar_grafico(df, eje_x='Temperature T(c)', columnas_y=['Weight (mg)','Heat Flow (Normalized) Q (W/g)','DTG_suavizado'], ruta_archivo = ruta_archivo, particiones = indices)
    """
    eje_x
    'Temperature T(c)'
    'Time t (min)'
    
    eje_y
    'Weight (mg)'
    'Weight (%)'
    'Heat Flow Q (mW)'
    'Heat Flow (Normalized) Q (W/g)'
    'DTG_suavizado'
    """
    guardar_graficas(figura, ruta_archivo)

    beta = int(search(r'VelocidadCalentamiento(\d+)', ruta_archivo).group(1))
    """
    Buscamos la palabra "VelocidadCalentamiento" seguida de uno o mas digitos (\d+),
    el parentesis captura dichos digitos. A continuacion se convierten a int dichos digitos almacenados en group
    Lo suyo seria ponerla en MAYUSCULA
    """
    resultados = aplicar_modelos(subgrupos, beta)

    calcular_energia(datos['heat_flow_q'],datos['time'])

    guardar_resultados(resultados, ruta_archivo)

    representar_regresion(resultados, ruta_archivo)

    return resultados


# ============================================================================
# FLUJO PRINCIPAL PARA EL ANALISIS MASICO DE LA BIOMASA
# ============================================================================
def main_biomasa (ruta_archivo):
    # Seleccion temperaturas para división en subgrupos (distintas fases de los componentes)
    temp_seleccionadas = {
        'hoja_enebro_15_aire': [115, 210, 530],
        'hoja_jara_15_aire': [140, 570],
        'hoja_pino_15_aire': [140, 220, 360, 570],
        'rama_enebro_15_aire': [120, 250, 520],
        'rama_jara_15_aire': [120, 255, 520],
        'rama_pino_15_aire': [130, 230, 530],
        'trozorama_pino_15_aire': [155, 220,600],
    }

    nombre_biomasa = os.path.splitext(os.path.basename(ruta_archivo))[0]
    temp_subgrupos = temp_seleccionadas[nombre_biomasa]

    df = cargar_datos(ruta_archivo)
    datos = procesar_datos(df)

    dtg = calcular_dtg(datos['temperature'], datos['weight_mg'])
    dtg_suavizado = suavizar_dtg(dtg, 400)
    df['DTG_suavizado'] = dtg_suavizado

    subgrupos, indices = dividir_en_subgrupos(datos['temperature'], datos['temperature_k'], datos['alpha'], datos['weight_mg'], datos['time'], datos['heat_flow_q'], temp_subgrupos)

    figura = generar_grafico(df, eje_x='Temperature T(c)', columnas_y=['Weight (mg)', 'Heat Flow (Normalized) Q (W/g)', 'DTG_suavizado'], ruta_archivo=ruta_archivo, particiones=indices)

    guardar_graficas(figura, ruta_archivo)

    resultados_biomasa = aplicar_modelos(subgrupos, beta = 15)

    guardar_resultados(resultados_biomasa, ruta_archivo)

    return resultados_biomasa

if __name__ == "__main__":
    # Directorio principal que contiene las carpetas con los archivos .csv
    directorio_componentes = 'DatosComponentes'

    parametros_componentes = {}

    # Leer el contenido del directorio 'DatosComponentes' solo una vez
    contenido_directorio_componentes = [(carpeta, os.listdir(os.path.join(directorio_componentes, carpeta)))
                            for carpeta in os.listdir(directorio_componentes)
                            if os.path.isdir(os.path.join(directorio_componentes, carpeta))]

    # Procesar los archivos de la lista obtenida en la lectura única del directorio
    for carpeta, archivos in contenido_directorio_componentes:
        ruta_carpeta = os.path.join(directorio_componentes, carpeta)

        # Recorrer los archivos en la lista de archivos de esa carpeta
        for archivo in archivos:
            # Verificar si el archivo es un archivo .csv
            if archivo.endswith('.csv'):
                ruta_archivo = os.path.join(ruta_carpeta, archivo)  # DatosComponentes/VelocidadCalentamiento5/celulosa_5_aire.csv, por ejemplo.
                # Procesar el archivo .csv
                nombre_componente = os.path.splitext(os.path.basename(ruta_archivo))[0]
                parametros_componentes[nombre_componente] = main_componentes(ruta_archivo)  # Modificarlo mas adelante para que cuando este trabajando con biomasa, pueda leer los resultados de un csv o excel directamente.

    # Directorio principal que contiene las carpetas con los archivos .csv
    directorio_biomasa = 'DatosBiomasa'

    # Creacion de diccionario para poder almacenar los resultados de cada Biomasa
    parametros_biomasa = {}

    # Leer el contenido del directorio 'DatosBiomasa' solo una vez
    contenido_directorio_biomasa = [(carpeta, os.listdir(os.path.join(directorio_biomasa, carpeta)))
                            for carpeta in os.listdir(directorio_biomasa)
                            if os.path.isdir(os.path.join(directorio_biomasa, carpeta))]

    # Procesar los archivos de la lista obtenida en la lectura única del directorio
    for carpeta, archivos in contenido_directorio_biomasa:
        ruta_carpeta = os.path.join(directorio_biomasa, carpeta)

        # Recorrer los archivos en la lista de archivos de esa carpeta
        for archivo in archivos:
            # Verificar si el archivo es un archivo .csv
            if archivo.endswith('.csv'):
                ruta_archivo = os.path.join(ruta_carpeta, archivo)  # DatosComponentes/VelocidadCalentamiento5/celulosa_5_aire.csv, por ejemplo.
                nombre_biomasa = os.path.splitext(os.path.basename(ruta_archivo))[0]
                # Obtencion porcentaje masico de los componentes puros en la biomasa
                parametros_biomasa[nombre_biomasa] = main_biomasa(ruta_archivo)  # Esto es algo parecido a crear una variable dinamica, python no permite crear variables dinamicas

    calculo_porcentajes_modelo1(parametros_biomasa,parametros_componentes)  # NO VALE
    #calculo_porcentajes_modelo2()