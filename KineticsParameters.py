# Development Branch. The work is done here, and then merged into the master branch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import odr_error
from sklearn.linear_model import LinearRegression
from re import search
import os.path

# ====================================================
# 1. CARGA DE DATOS Y PROCESAMIENTO INICIAL
# ====================================================

# Cargar el archivo CSV
csv_path = 'Datos/VelocidadCalentamiento5/celulosa_5_aire.csv'
df = pd.read_csv(csv_path, sep=';')  # Leer el archivo CSV

# Reemplazar las comas por puntos en los valores numéricos
df['Temperature T(c)'] = df['Temperature T(c)'].str.replace(',', '.').astype(float)
df['Time t (min)'] = df['Time t (min)'].str.replace(',', '.').astype(float)
df['Weight (mg)'] = df['Weight (mg)'].str.replace(',', '.').astype(float)
df['Weight (%)'] = df['Weight (%)'].str.replace(',', '.').astype(float)
df['Heat Flow Q (mW)'] = df['Heat Flow Q (mW)'].str.replace(',', '.').astype(float)
df['Heat Flow (Normalized) Q (W/g)'] = df['Heat Flow (Normalized) Q (W/g)'].str.replace(',', '.').astype(float)

# Convertir las columnas de Pandas a arrays de Numpy
time = np.array(df['Time t (min)'])
temperature = np.array(df['Temperature T(c)'])
weight_percent = np.array(df['Weight (%)'])
weight_mg = np.array(df['Weight (mg)'])
heat_flow_q = np.array(df['Heat Flow Q (mW)'])
heat_flow_normalized = np.array(df['Heat Flow (Normalized) Q (W/g)'])

# Conversion a Kelvin
temperature_K = temperature  + 273

# Calcular fraccion de conversion (alpha), asegurándonos de que no haya valores de alpha fuera del rango esperado
weight_0 = weight_mg[0]     # Masa inicial
weight_f = weight_mg[-1]    # Masa final
alpha = np.clip((weight_0 - weight_mg) / (weight_0 - weight_f), 0, 1)

# Evitar valores problemáticos cercanos a 0 o 1
valid_alpha_mask = (alpha > 0) & (alpha < 1)  # Creacion de una mascara para filtrar solo los datos validos

# Aplicacion del filtro a los arrays de temperatura y alpha. Son los que intervienen en el ajuste lineal por CR
temperature_K = temperature_K[valid_alpha_mask]
alpha = alpha[valid_alpha_mask]

# Breve comprobacion.
hay_negativos = np.any (alpha<=0)
print(f"Hay numeros negativos, {hay_negativos}")
mayores_que_uno = np.any (alpha>=1)
print(f"Hay numeros positivos, {mayores_que_uno}")


# ====================================================
# 2. OBTENCION CURVA DTG Y SUAVIZADO DE LA MISMA
# ====================================================

# Obtención de la curva DTG usando metodo numerico de diferencias finitas
dtg = np.zeros_like(weight_mg)  # Crear un array vacío para la DTG del mismo tamaño que el dado, en este caso; weight_mg
for i in range(1, len(temperature) - 1):
    delta_temp = temperature[i + 1] - temperature[i - 1]  # Diferencia de temperatura entre puntos

    # Evitar división por cero verificando si la diferencia de temperatura es mayor que un umbral
    if delta_temp != 0:
        dtg[i] = (weight_mg[i + 1] - weight_mg[i - 1]) / delta_temp
    else:
        dtg[i] = 0  # Si hay división por cero, asignar 0 o un valor que elijas

# En los extremos (primer y último punto), se usa la diferencia hacia adelante y hacia atrás
dtg[0] = (weight_mg[1] - weight_mg[0]) / (temperature[1] - temperature[0]) if (temperature[1] - temperature[0]) != 0 else 0
dtg[-1] = (weight_mg[-1] - weight_mg[-2]) / (temperature[-1] - temperature[-2]) if (temperature[-1] - temperature[-2]) != 0 else 0

# Reemplazar valores infinitos o NaN en la curva DTG para evitar errores en la gráfica
dtg = np.nan_to_num(dtg, nan=0.0, posinf=0.0, neginf=0.0)

#Funcion para eliminar el ruido mediante la técnica del promedio movil (Moving Average)
def moving_average(data, window_size):
    # Crear un array de promedios móviles, usando una ventana deslizante
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

# Aplicar el suavizado a la curva DTG - Cuanto mayor sea window_size, mayor suavizada sera la curva y menos precisos son los valores de DTG
dtg_suavizado = moving_average(dtg, 400)
df['DTG_suavizado'] = dtg_suavizado #Agregar dtg_suavizado al Dataframe para su selección


# ====================================================
# 3. CREAR SUBGRUPOS BASADOS EN TEMPERATURAS
# ====================================================

# Función para encontrar el número más cercano en el array de Temperature. Necesario ya que no siempre la temperatura que introduce el usuario existe en el array de temperaturas
def encontrar_mas_cercano(arr,num):
    idx = (np.abs(arr - num)).argmin()  # Encuentra el índice del valor más cercano
    return arr[idx], idx  # Devuelve el valor más cercano y su índice

# Introducir los valores de las temperaturas donde se quiera subdividir la curva de TGA.
temp_seleccionadas = input("Introduce las temperaturas deseadas para crear distintas zonas en el TGA (separados por comas): ")
temp_seleccionadas = [float(n) for n in temp_seleccionadas.split(",")]      # Conversion del string leido a numeros float


# Encontrar los índices de las temperaturas seleccionadas o las más cercanas
indices = []
for num in temp_seleccionadas:
    temp_cercana, idx = encontrar_mas_cercano(temperature,num)
    print(f"La temperatura más cercana a {num:.2f} en el vector de Temperatura es {temp_cercana:.2f} en el índice {idx}")
    indices.append(idx)     # Añade el indice idx (el indice de la temp mas cercana) a lista de indices

# Ordenar los índices por si no están en orden y eliminar duplicados
indices = sorted(set(indices))

# Dividir los arrays en subgrupos según esos índices
subgrupos_temp = np.split(temperature, indices)
subgrupos_tempK = np.split(temperature_K, indices)
subgrupos_alpha = np.split(alpha, indices)
subgrupos_weight = np.split(weight_mg, indices)

# ====================================================
# 4. FUNCIONES CINÉTICAS Y AJUSTE DE MODELOS
# ====================================================

R = 8.3144  # Constante de los gases en J/mol·K
beta = int(search(r'VelocidadCalentamiento(\d+)', csv_path).group(1))        # Buscamos la palabra "VelocidadCalentamiento" seguida de uno o mas digitos (\d+),
                                                                                    # el parentesis captura dichos digitos. A continuacion se convierten a int dichos digitos almacenados en group
                                                                                    # Lo suyo seria ponerla en MAYUSCULA.
# Funciones g(alpha) para los distintos modelos cinéticos
# Chemical reaction model
# Zero order (CR0)
def cr0(alpha): return alpha
# First order (CR1)
def cr1(alpha): return -np.log(1 - alpha)
# Second order (CR2)
def cr2(alpha): return 2*((1 - alpha)**(-1.5) - 1)
# Third order (CR3)
def cr3(alpha): return 0.5*((1 - alpha)**(-2)- 1)

# Diffusion model
# Parabolic law (DM1)
def dm1(alpha): return alpha**2
# Valensi (2D diffusion) (DM2)
def dm2(alpha): return alpha + (1 - alpha)*np.log(1 - alpha)
# Ginstling - Broushtein (DM3)
def dm3(alpha): return (1 - (2*alpha/3)) - (1 - alpha)**(2/3)

#Nucleation and growth model
# Avrami - Erofeev (n = 1.5) (NG1.5)
def ng1_5(alpha): return (-np.log(1 - alpha))**(2/3)
# Avrami - Erofeev (n = 2) (NG2)
def ng2(alpha): return (-np.log(1 - alpha))**(1/2)
# Avrami - Erofeev (n = 3) (NG3)
def ng3(alpha): return (-np.log(1 - alpha))**(1/3)


# Función para ajustar el modelo CR
def ajustar_modelo_cr(temperature, alpha, g_alpha_func):
    T_inv = 1 / temperature  # 1/T (T en Kelvin)
    g_alpha = g_alpha_func(alpha)

    # Variables para la regresión, LinearRegression espera que los datos de entrada x (variables independientes) tengan
    # dos dimensiones, donde cada fila representa un punto de datos y cada columna una variable. Por lo tanto, reshape
    # transforma el vector 1D en un matriz con una sola columna.
    x = T_inv.reshape(-1, 1)
    y = np.log(g_alpha / temperature ** 2)

    # Ajuste lineal con sklearn
    reg = LinearRegression()
    reg.fit(x, y)

    # Parámetros del ajuste
    pendiente = reg.coef_[0]
    intercepto = reg.intercept_
    r2 = reg.score(x, y)

    # Calcular energía de activación (Ea) y factor pre-exponencial (A)
    Ea = -pendiente * R  # Energía de activación en J/mol
    A = (beta*Ea/R)* np.exp(intercepto)  # Factor pre-exponencial, realmente sería A = (e^intercepto) * (beta*Ea/R). Pero se simplifica ya que estas son constantes.

    return Ea, A, r2


# ====================================================
# 5. GUARDAR RESULTADOS EN EXCEL
# ====================================================

# Aplicar el ajuste para cada subgrupo y modelo
modelos = {
    "CR0": cr0,
    "CR1": cr1,
    "CR2": cr2,
    "CR3": cr3,
    "DM1": dm1,
    "DM2": dm2,
    "DM3": dm3,
    "NG1.5": ng1_5,
    "NG2": ng2,
    "NG3": ng3,
}
# Almacenar resultados en una lista
resultados = []     # Crear un diccionario de DataFrames donde se almacenarán los datos de cada subgrupo
for i, (temp_sub, alpha_sub) in enumerate(zip(subgrupos_tempK, subgrupos_alpha)):   # Iterar sobre cada subgrupo y ajustar todos los modelos
    for nombre_modelo, modelo_func in modelos.items():
        Ea, A, r2 = ajustar_modelo_cr(temp_sub, alpha_sub, modelo_func)
        # Añadir los resultados a la lista de DataFrames
        resultados.append({
            "Model": f"{nombre_modelo} Subgroup {i+1}",
            "Ea (J/mol)": Ea,
            "A (1/s)": A,
            "R²": r2
        })

# Convertir la lista de resultados a un DataFrame de Pandas
df_resultados = pd.DataFrame(resultados)

# Guardar el DataFrame en un archivo Excel, si se quieren crear varios archivos excel simplemente cambiar nombre archivo aqui antes de ejecutar
output_excel_file = 'parametros_cineticos.xlsx'
df_resultados.to_excel(output_excel_file, index=False)

# Confirmación de guardado exitoso
print(f"Los resultados se han guardado en '{output_excel_file}'")


# ====================================================
# 6. GENERAR GRÁFICOS
# ====================================================

# Función para generar gráficos con múltiples ejes Y y particiones
def generar_grafico(eje_x, columnas_y, title, particiones=None):
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


# Ejemplo de uso con columnas seleccionadas y ordenadas manualmente
generar_grafico(eje_x='Temperature T(c)', columnas_y=['Weight (mg)','Heat Flow (Normalized) Q (W/g)','DTG_suavizado'], title=os.path.splitext(csv_path.split('/')[-1])[0], particiones=indices)
# Para obtener el titulo a partir del nombre del archivo csv; primero se ha divido el nombre del archivo csv usando las '/' que hay en el mismo nombre, se ha seleccionado la ultima palabra [-1]
# finalmente se ha dividido "celulosa_5_aire" y ".csv", quedandonos con la primera [0]

#Muchisimo cuidado al meter el nombre, debe ser exactamente igual. Copiarlo de aqui.
#eje_x
#'Temperature T(c)'
#'Time t (min)'

#eje_y
#'Weight (mg)'
#'Weight (%)'
#'Heat Flow Q (mW)'
#'Heat Flow (Normalized) Q (W/g)'
#'DTG_suavizado'

# ====================================================
# 6. REPRESENTACION DE LAS REGRESIONES
# ====================================================
# Se podria dividir en 2 o 3 funciones. La primera seria la de solicitar al usuario que introduzca subgrupo y modelo. La siguiente seria la de trabajar con cada subgrupo y la ultima seria la de representar
while True:
    subgrupo_reg = input("Introduce el subgrupo deseado (1 - 3) o 'exit' para salir: ")
    if subgrupo_reg.lower() == "exit":
        print ("Saliendo del proceso.")
        break

    modelo_reg = input ("Introduce el modelo deseado o 'exit' para salir: ").upper()  # Modificarlo para que el usuario nunca cometa errores, mostrando una lista por ejemplo
    if modelo_reg.lower() == "exit":
        print("Saliendo del proceso.")
        break

    try:
        subgrupo_reg = int(subgrupo_reg)
        if subgrupo_reg not in [1,2,3]:
            print("Por favor, introduce un subgrupo valido (1 - 3): ")
            continue

        subgrupo_TempK_reg = subgrupos_tempK[subgrupo_reg - 1]
        subgrupo_alpha_reg = subgrupos_alpha[subgrupo_reg - 1]
        subgrupo_weight_reg = subgrupos_weight[subgrupo_reg - 1]
        subgrupo_temp_reg = subgrupos_temp [subgrupo_reg - 1]
        modelo_func_reg = modelos[modelo_reg]

        Ea_reg, A_reg, r2_reg = ajustar_modelo_cr(subgrupo_TempK_reg,subgrupo_alpha_reg,modelo_func_reg)
        print (f"Modelo: {modelo_reg}\nEa: {Ea_reg}\nA: {A_reg}\nr2: {r2_reg}")

        reg_x = 1 / subgrupo_TempK_reg
        reg_y = np.log (modelo_func_reg(subgrupo_alpha_reg) / subgrupo_TempK_reg ** 2)

        coeficientes = np.polyfit(reg_x,reg_y,1)
        pendiente, ordenada = coeficientes              # De aqui puedo extraer de nuevo Ea y A y obviamente se corresponden con los extraidos previamente. El problema esta en que al
                                                        # representar con los obtenidos previamente me da error.

        ajuste_reg = pendiente * (1/subgrupo_TempK_reg) + ordenada
        ajuste_reg1 = np.log(A_reg*R / beta*Ea_reg) + (reg_x * Ea_reg/R)

        # Representación
        fig, (ax1, ax2) = plt.subplots (2)

        ax1.plot (subgrupo_temp_reg, subgrupo_weight_reg, linestyle='-', color='b', label='Peso vs. Temperatura')
        ax1.set_xlabel('Temperatura (°C)')
        ax1.set_ylabel('Peso (mg)')
        ax1.set_title ('Gráfico de Temperatura vs. Peso')
        ax1.legend()

        ax2.plot(reg_x, reg_y, linestyle='--', color='g', label='1/T vs ln(g(alpha) / T^2)')
        #ax2.plot(reg_x, ajuste_reg,linestyle='-', color='r', label='Ajuste lineal')
        ax2.plot(reg_x, ajuste_reg1, linestyle='-', color='b', label='Ajuste lineal2')
        ax2.set_xlabel('1/T')
        ax2.set_ylabel('ln(g(alpha) / T^2)')
        ax2.set_title('Expresion Arrhenius linealizada')
        ax2.legend()

        # Mostrar la gráfica
        fig.tight_layout()
        fig.suptitle("Regresión lineal")
        plt.show()


    except ValueError:
        print ("Entrada invalida. Asegurate de introducir un numero para el numero de subrgrupo")


