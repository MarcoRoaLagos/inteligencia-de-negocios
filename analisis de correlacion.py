import pandas as pd
from collections import Counter
from itertools import combinations

def cargar_datos(ruta_archivo, columnas):
    df = pd.read_excel(ruta_archivo)
    return df[columnas].dropna()

def obtener_transacciones(df):
    return df.apply(lambda fila: set(fila.values), axis=1).tolist()

def contar_items_y_combinaciones(transacciones, combo_size=2):
    contador_items = Counter()
    contador_combos = Counter()
    for trans in transacciones:
        # Contar items individuales
        contador_items.update(trans)
        # Contar combinaciones de tamaño combo_size
        combos = combinations(sorted(trans), combo_size)
        contador_combos.update(combos)
    return contador_items, contador_combos

def calcular_metricas(contador_items, contador_combos, n_transacciones):
    reglas = []
    for (item1, item2), freq in contador_combos.items():
        soporte = freq / n_transacciones
        confianza_1_2 = freq / contador_items[item1]  # confianza de item1 -> item2
        confianza_2_1 = freq / contador_items[item2]  # confianza de item2 -> item1
        lift_1_2 = confianza_1_2 / (contador_items[item2] / n_transacciones)
        lift_2_1 = confianza_2_1 / (contador_items[item1] / n_transacciones)
        
        # Guardamos ambas reglas direccionales (item1->item2 y item2->item1)
        reglas.append({
            'Regla': f"{item1} -> {item2}",
            'Soporte': soporte,
            'Confianza': confianza_1_2,
            'Lift': lift_1_2
        })
        reglas.append({
            'Regla': f"{item2} -> {item1}",
            'Soporte': soporte,
            'Confianza': confianza_2_1,
            'Lift': lift_2_1
        })
    return reglas

def filtrar_reglas(reglas, soporte_min=0.1, confianza_min=0.6, lift_min=1.2):
    return [r for r in reglas if r['Soporte'] >= soporte_min and
                                  r['Confianza'] >= confianza_min and
                                  r['Lift'] >= lift_min]

def imprimir_reglas(reglas):
    if not reglas:
        print("No se encontraron reglas que cumplan los criterios.")
        return
    print("Reglas de asociación filtradas:\n")
    for r in reglas:
        print(f"{r['Regla']}: Soporte={r['Soporte']:.2f}, Confianza={r['Confianza']:.2f}, Lift={r['Lift']:.2f}")

if __name__ == "__main__":
    columnas_interes = ['descripcion', 'negocio', 'mes', 'linea']
    df_filtrado = cargar_datos("mermas 1.xlsx", columnas_interes)
    transacciones = obtener_transacciones(df_filtrado)
    n_trans = len(transacciones)
    
    contador_items, contador_combos = contar_items_y_combinaciones(transacciones)
    reglas = calcular_metricas(contador_items, contador_combos, n_trans)
    
    reglas_filtradas = filtrar_reglas(reglas, soporte_min=0.1, confianza_min=0.6, lift_min=1.2)
    imprimir_reglas(reglas_filtradas)
