# Development Branch. The work is done here, and then merged into the master branch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from re import search
import os.path

# Constantes globales
R = 8.3144              # Constante de los gases en J/mol·K

# ====================================================
# 1. CARGA Y PROCESAMIENTO DE DATOS
# ====================================================

def cargar_datos(ruta_archivo):
    """
    # Se cargan los datos del archivo .csv a un dataframe (df) y se sustituyen las ',' que hay en los valores de las columnas por '.'
    # Ademas se convierten dichos valores a float y se devuelve el df con las modificaciones.
    """
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

    # Calcular fracción de conversión (alpha)
    weight_0, weight_f = weight_mg[0], weight_mg[-1]
    alpha = np.clip((weight_0 - weight_mg) / (weight_0 - weight_f), 0, 1)

    # Filtrar valores de alpha válidos
    valid_alpha_mask = (alpha > 0) & (alpha < 1)

    # Devuelve un diccionario que alverga los datos filtrados
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
    dtg = np.zeros_like(weight_mg)      # Crear un array vacío para la DTG del mismo tamaño que el dado, en este caso; weight_mg
    for i in range(1, len(temperature) - 1):
        # Evitar división por cero
        delta_temp = temperature[i + 1] - temperature[i - 1]
        dtg[i] = (weight_mg[i + 1] - weight_mg[i - 1]) / delta_temp if delta_temp != 0 else 0

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
        print(
            f"La temperatura más cercana a {temp:.2f} en el vector de Temperatura es {temp_cercana:.2f} en el índice {idx}")
        indices.append(idx)  # Añade el indice idx (el indice de la temp mas cercana) a lista de indices

    # Ordenar los índices por si no están en orden y eliminar duplicados
    indices = sorted(set(indices))

    # Dividir los arrays en subgrupos según esos índices
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
    A = (beta * Ea / R) * np.exp(intercepto)    # Factor pre-exponencial

    return Ea, A, r2


def aplicar_modelos(subgrupos, beta):
    # Almacenar resultados en una lista
    resultados = []     # Creacion de una lista vacia para almacenar los resultados de los modelos ajustados en cada subgrupo
    for i, (temp_K_sub, alpha_sub) in enumerate(zip(subgrupos['temp_K'], subgrupos['alpha'])): # Iterar sobre cada subgrupo y ajustar todos los modelos
        for nombre, modelo_func in modelos.items():
            Ea, A, r2 = ajustar_modelo(temp_K_sub, alpha_sub, modelo_func, beta)
            # Creacion de un diccionario con los resultados del ajuste y agregacion de dicho diccionario a la lista resultados
            resultados.append({"Model": f"{nombre} Subgroup {i+1}", "Ea (J/mol)": Ea, "A (1/s)": A, "R²": r2})
    return pd.DataFrame(resultados) # Se convierte la lista 'resultados' en un dataframe de Pandas.

# ====================================================
# 6. GUARDAR RESULTADOS Y GRAFICAR
# ====================================================
def guardar_resultados(resultados, output_excel_file='parametros_cineticos.xlsx'):
    resultados.to_excel(output_excel_file, index=False)
    print(f"Resultados guardados en '{output_excel_file}'")

# Función para generar gráficos con múltiples ejes Y y particiones
def generar_grafico(df, eje_x, columnas_y, title, particiones=None):
    if not columnas_y:
        print("No se seleccionaron columnas para el eje Y. Por favor, especifica al menos una columna.")
        return

    # Comprobar si las columnas seleccionadas existen en el DataFrame
    columnas_validas = [col for col in columnas_y if col in df.columns]
    if not columnas_validas:
        print(f"Ninguna de las columnas seleccionadas existe en el DataFrame. Columnas disponibles: {list(df.columns)}")
        return

    fig, ax1 = plt.subplots()

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
    fig.suptitle(f"Gráfico de {title}")
    fig.tight_layout()
    plt.show()


# ====================================================
# 7. ENERGÍA ASOCIADA AL PROCESO
# ====================================================
def calcular_energia(heat_flow_q, time):
    heat_flow_w = heat_flow_q * 0.001
    time_s = time * 60
    energia = np.trapezoid(heat_flow_w, time_s)
    return energia

# ====================================================
# 7. REPRESENTACION DE LAS REGRESIONES
# ====================================================
# ====================================================
# Función para solicitar los datos del usuario
# ====================================================
def solicitar_datos():
    while True:
        subgrupo_reg = input("Introduce el subgrupo deseado (1 - 3) o 'exit' para salir: ")
        if subgrupo_reg.lower() == "exit":
            print("Saliendo del proceso.")
            return None, None

        try:
            subgrupo_reg = int(subgrupo_reg)
            if subgrupo_reg not in [1, 2, 3]:
                print("Por favor, introduce un subgrupo válido (1 - 3): ")
                continue

            modelos_validos = ["CR0", "CR1", "CR2", "CR3", "DM0", "DM1", "DM2", "NG1.5", "NG2", "NG3"]
            print(f"Modelos disponibles: {', '.join(modelos_validos)}")
            modelos_reg = input("Introduce tres modelos separados por comas: ").upper().split(',')

            if "EXIT" in modelos_reg:
                print("Saliendo del proceso.")
                return None, None

            # Validar que se hayan introducido exactamente 3 modelos válidos
            if len(modelos_reg) != 3 or any(modelo.strip() not in modelos_validos for modelo in modelos_reg):
                print("Por favor, introduce tres modelos válidos de la lista.")
                continue

            return subgrupo_reg, [modelo.strip() for modelo in modelos_reg]     # Devolver lista de nombres eliminando los espacios si existieran

        except ValueError:
            print("Entrada inválida. Asegúrate de introducir un número para el número de subgrupo.")


# ====================================================
# Función para realizar los cálculos de cada modelo
# ====================================================
def calcular_regresion(subgrupo_reg, modelos_reg, subgrupos, beta):
    subgrupo_tempk_reg = subgrupos['temp_K'][subgrupo_reg - 1]
    subgrupo_alpha_reg = subgrupos['alpha'][subgrupo_reg - 1]
    subgrupo_weight_reg = subgrupos['weight'][subgrupo_reg - 1]
    subgrupo_temp_reg = subgrupos['temp'][subgrupo_reg - 1]

    subgrupo_time_reg = subgrupos['time'][subgrupo_reg - 1]     # En realidad no deberia llamarse, subgrupo_time_reg, pq no hace nada para la regresion. Este subgrupo de tiempo se usa para calcular
                                                                # la integral de DSC temporal en ese tramo unicamente.
    subgrupo_heat_flow_q_reg = subgrupos['heat_flow_q'][subgrupo_reg - 1]

    resultados_reg = {}

    for modelo in modelos_reg:
        modelo_func_reg = modelos[modelo]

        # Calcular los parámetros de la regresión para el modelo
        Ea_reg, A_reg, r2_reg = ajustar_modelo(subgrupo_tempk_reg, subgrupo_alpha_reg, modelo_func_reg, beta)
        print(f"Modelo: {modelo}\nEa: {Ea_reg}\nA: {A_reg}\nr2: {r2_reg}")

        # Calcular los valores para la representación de la regresión y su ajuste lineal
        reg_x = 1 / subgrupo_tempk_reg
        reg_y = np.log(modelo_func_reg(subgrupo_alpha_reg) / subgrupo_tempk_reg ** 2)

        coeficientes = np.polyfit(reg_x, reg_y, 1)
        pendiente, ordenada = coeficientes                  # De aqui puedo extraer de nuevo Ea y A y obviamente se corresponden con los extraidos previamente. El problema esta en que al
                                                            # representar con los obtenidos previamente me da error y es por eso por lo que se realiza el ajuste lineal de nuevo aqui

        ajuste_reg = pendiente * (1 / subgrupo_tempk_reg) + ordenada

        resultados_reg[modelo] = {
            "reg_x": reg_x,
            "reg_y": reg_y,
            "ajuste_reg": ajuste_reg,
            "Ea_reg" : Ea_reg,
            "A_reg" : A_reg,
            "R2_reg": r2_reg
        }

    energia_subgrupo = np.trapezoid(subgrupo_heat_flow_q_reg * 0.001, subgrupo_time_reg * 60)
    return subgrupo_temp_reg, subgrupo_weight_reg, resultados_reg, energia_subgrupo

# ====================================================
# Función para la representación gráfica
# ====================================================
def representar_regresion(subgrupo_temp_reg, subgrupo_weight_reg, resultados_reg):
    # Crear la figura con una cuadrícula de 3 filas x 2 columnas
    fig, axs = plt.subplots(2, 3, figsize=(12, 10))

    # Desactivar los ejes que no se van a usar (posición 0,0 y 0,2)
    axs[0, 0].axis('off')
    axs[0, 2].axis('off')

    # Gráfica de temperatura vs. peso en la posición 0,1 (segunda columna de la primera fila)
    axs[0, 1].plot(subgrupo_temp_reg, subgrupo_weight_reg, linestyle='-', color='b', label='Peso vs. Temperatura')
    axs[0, 1].set_xlabel('Temperatura (°C)')
    axs[0, 1].set_ylabel('Peso (mg)')
    axs[0, 1].set_title('Gráfico de Temperatura vs. Peso')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Gráficas de regresión lineal para cada modelo en la segunda fila (1,0) (1,1) y (1,2)
    for idx, (modelo, datos) in enumerate(resultados_reg.items()):
        reg_x = datos["reg_x"]
        reg_y = datos["reg_y"]
        ajuste_reg = datos["ajuste_reg"]
        Ea = datos["Ea_reg"]
        A = datos ["A_reg"]
        R2 = datos ["R2_reg"]

        # Determinar la posición de la gráfica en la fila inferior
        axs[1 , idx].plot(reg_x, reg_y, linestyle='--', label=f'{modelo} - ln(g(alpha) / T^2)')
        axs[1 , idx].plot(reg_x, ajuste_reg, linestyle='-', label=f'{modelo} - Ajuste lineal\nEa = {Ea:.2f} kJ/mol\nA = {A:.2e} s⁻¹\nr² = {R2:.4f}')

        # Configurar cada gráfica de la fila inferior
        axs[1, idx].set_xlabel('1/T')
        axs[1, idx].set_ylabel('ln(g(alpha) / T^2)')
        axs[1, idx].set_title(f'{modelo}')
        axs[1, idx].grid(True)
        axs[1, idx].legend(loc = "best", fontsize = "small")

    # Ajustar el diseño para que no haya superposición de textos
    fig.tight_layout()
    fig.suptitle("Regresiones lineales y Gráfico de Temperatura vs. Peso", y=1.02)
    plt.show()


# ====================================================
# FLUJO PRINCIPAL
# ====================================================
def main(ruta_archivo, temp_subgrupos):
    df = cargar_datos(ruta_archivo)
    datos = procesar_datos(df)

    dtg = calcular_dtg(datos['temperature'], datos['weight_mg'])
    dtg_suavizado = suavizar_dtg(dtg, 400)
    df['DTG_suavizado'] = dtg_suavizado

    subgrupos, indices = dividir_en_subgrupos(datos['temperature'], datos['temperature_k'], datos['alpha'],
                                              datos['weight_mg'], datos['time'], datos['heat_flow_q'],
                                              temp_subgrupos)
    generar_grafico(df, eje_x='Temperature T(c)', columnas_y=['Weight (mg)','Heat Flow (Normalized) Q (W/g)','DTG_suavizado'], title=os.path.splitext(ruta_archivo.split('/')[-1])[0], particiones = indices)
    """Para obtener el titulo a partir del nombre del archivo csv; primero se ha divido el nombre del archivo csv usando las '/' que hay en el mismo nombre, se ha seleccionado la ultima palabra [-1]
    finalmente se ha dividido "celulosa_5_aire" y ".csv", quedandonos con la primera [0]
    Muchisimo cuidado al meter el nombre, debe ser exactamente igual. Copiarlo de aqui.
    eje_x
    'Temperature T(c)'
    'Time t (min)'
    
    eje_y
    'Weight (mg)'
    'Weight (%)'
    'Heat Flow Q (mW)'
    'Heat Flow (Normalized) Q (W/g)'
    'DTG_suavizado'"""

    beta = int(search(r'VelocidadCalentamiento(\d+)', ruta_archivo).group(1))
    """Buscamos la palabra "VelocidadCalentamiento" seguida de uno o mas digitos (\d+),
        el parentesis captura dichos digitos. A continuacion se convierten a int dichos digitos almacenados en group
        Lo suyo seria ponerla en MAYUSCULA"""
    resultados = aplicar_modelos(subgrupos, beta)


    energia_total = calcular_energia(datos['heat_flow_q'],datos['time'])
    print(f"La energia obtenida de la integral temporal de la curva DSC es: {energia_total: .4f} J")

    guardar_resultados(resultados)

    while True:
        subgrupo_reg, modelos_reg = solicitar_datos()
        if subgrupo_reg is None or modelos_reg is None:
            break
        subgrupo_temp_reg, subgrupo_weight_reg, resultados_reg, energia_subgrupo = calcular_regresion(subgrupo_reg,
                                                                                                      modelos_reg, subgrupos, beta)
        print(f"\nLa energia asociada a dicho proceso es: {energia_subgrupo: .4f} J")
        representar_regresion(subgrupo_temp_reg, subgrupo_weight_reg, resultados_reg)


if __name__ == "__main__":
    # Ruta al archivo CSV
    csv_path = "Datos/VelocidadCalentamiento30/xilano_30_n2.csv"

    # Lista de temperaturas seleccionadas para dividir los datos
    temp_seleccionadas = [220, 340]

    # Inicializar el flujo principal
    main(csv_path, temp_seleccionadas)



