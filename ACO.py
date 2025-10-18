"""
ACO.py
Implementación simple del algoritmo Ant Colony Optimization (ACO) para el problema
del viajante de comercio (TSP). Comentarios en español y ejemplo ejecutable.

Requisitos: numpy, matplotlib
Instalación: pip install numpy matplotlib
Ejecutar: python ACO.py
"""
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Parámetros por defecto (ajustables desde la línea principal)
NUM_HORMIGAS = 30        # número de hormigas por iteración
NUM_ITERACIONES = 200    # número de iteraciones
EVAPORACION = 0.5        # tasa de evaporación de feromona (0..1)
ALFA = 1.0               # influencia de la feromona
BETA = 5.0               # influencia de la heurística (1/distancia)
Q = 100.0                # cantidad de feromona depositada por solución (constante)

def distancia_euclidiana(ciudad_a, ciudad_b):
    """Distancia euclidiana entre dos puntos (x,y)."""
    return math.hypot(ciudad_a[0] - ciudad_b[0], ciudad_a[1] - ciudad_b[1])

def matriz_distancias(ciudades):
    """Construye matriz simétrica de distancias entre ciudades."""
    n = len(ciudades)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = distancia_euclidiana(ciudades[i], ciudades[j])
            D[i, j] = d
            D[j, i] = d
    return D

def inicializar_feromonas(n, tau0=1.0):
    """Crea matriz de feromona inicial con valor tau0."""
    return np.ones((n, n)) * tau0

def probabilidad_seleccion(origen, no_visitadas, feromona, dist, alfa, beta):
    """Calcula distribución de probabilidad para elegir la siguiente ciudad."""
    numeradores = []
    for j in no_visitadas:
        tau = feromona[origen, j] ** alfa
        eta = (1.0 / dist[origen, j]) ** beta if dist[origen, j] > 0 else 1e6
        numeradores.append(tau * eta)
    suma = sum(numeradores)
    if suma == 0:
        # fallback: selección uniforme
        probs = [1.0 / len(no_visitadas)] * len(no_visitadas)
    else:
        probs = [num / suma for num in numeradores]
    return probs

def construir_ruta(inicio, feromona, dist, alfa, beta):
    """Construye una ruta completa iniciando en 'inicio' siguiendo probabilidades."""
    n = dist.shape[0]
    ruta = [inicio]
    visitadas = set(ruta)
    actual = inicio
    while len(ruta) < n:
        candidatos = [j for j in range(n) if j not in visitadas]
        probs = probabilidad_seleccion(actual, candidatos, feromona, dist, alfa, beta)
        siguiente = random.choices(candidatos, weights=probs, k=1)[0]
        ruta.append(siguiente)
        visitadas.add(siguiente)
        actual = siguiente
    return ruta

def longitud_ruta(ruta, dist):
    """Calcula la longitud total de una ruta (ciclo cerrado, vuelve al inicio)."""
    total = 0.0
    for i in range(len(ruta)):
        a = ruta[i]
        b = ruta[(i + 1) % len(ruta)]
        total += dist[a, b]
    return total

def depositar_feromona(feromona, soluciones, dist, Q):
    """Actualiza depósito de feromona en función de soluciones (lista de rutas)."""
    delta = np.zeros_like(feromona)
    for ruta in soluciones:
        L = longitud_ruta(ruta, dist)
        contrib = Q / L if L > 0 else 0
        for i in range(len(ruta)):
            a = ruta[i]
            b = ruta[(i + 1) % len(ruta)]
            delta[a, b] += contrib
            delta[b, a] += contrib  # feromona simétrica
    feromona += delta

def evaporar_feromona(feromona, rho):
    """Evaporación multiplicativa de la feromona."""
    feromona *= (1.0 - rho)

def aco_tsp(ciudades, num_hormigas=NUM_HORMIGAS, num_iter=NUM_ITERACIONES,
            rho=EVAPORACION, alfa=ALFA, beta=BETA, Q_const=Q, mostrar_progreso=True):
    """
    Implementación principal del ACO para TSP.
    Devuelve la mejor ruta encontrada y su longitud.
    """
    n = len(ciudades)
    dist = matriz_distancias(ciudades)
    feromona = inicializar_feromonas(n, tau0=1.0)
    mejor_ruta = None
    mejor_long = float('inf')
    historial_mejores = []

    for it in range(num_iter):
        soluciones = []
        for k in range(num_hormigas):
            inicio = random.randrange(n)
            ruta = construir_ruta(inicio, feromona, dist, alfa, beta)
            soluciones.append(ruta)
            L = longitud_ruta(ruta, dist)
            if L < mejor_long:
                mejor_long = L
                mejor_ruta = ruta.copy()
        # evaporar y depositar feromona
        evaporar_feromona(feromona, rho)
        depositar_feromona(feromona, soluciones, dist, Q_const)
        historial_mejores.append(mejor_long)
        if mostrar_progreso and (it % max(1, num_iter // 10) == 0 or it == num_iter - 1):
            print(f"Iter {it+1}/{num_iter} - mejor longitud: {mejor_long:.4f}")

    return mejor_ruta, mejor_long, historial_mejores

def generar_ciudades_aleatorias(n, seed=None, escala=100):
    """Genera n ciudades con coordenadas aleatorias en un cuadrado [0,escala]."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    return [(random.uniform(0, escala), random.uniform(0, escala)) for _ in range(n)]

def dibujar_ruta(ciudades, ruta, titulo=None):
    """Dibuja las ciudades y la ruta encontrada (ciclo cerrado)."""
    xs = [ciudades[i][0] for i in ruta] + [ciudades[ruta[0]][0]]
    ys = [ciudades[i][1] for i in ruta] + [ciudades[ruta[0]][1]]
    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, '-o')
    for idx, (x, y) in enumerate(ciudades):
        plt.text(x, y, str(idx))
    if titulo:
        plt.title(titulo)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Ejemplo: 20 ciudades aleatorias
    N_CIUDADES = 20
    ciudades = generar_ciudades_aleatorias(N_CIUDADES, seed=42, escala=100)
    mejor_ruta, mejor_long, historial = aco_tsp(ciudades,
                                               num_hormigas=40,
                                               num_iter=200,
                                               rho=0.5,
                                               alfa=1.0,
                                               beta=5.0,
                                               Q_const=100.0,
                                               mostrar_progreso=True)
    print("\nMejor longitud final:", mejor_long)
    print("Mejor ruta (orden de índices):", mejor_ruta)
    dibujar_ruta(ciudades, mejor_ruta, titulo=f"Mejor ruta - longitud {mejor_long:.2f}")

