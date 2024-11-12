import numpy as np
from rich import print
import time
from itertools import product
import pandas as pd
from tqdm import tqdm
import math
from utils import *

class AntColonyOptimizer:
    def __init__(self, instance_data, weight_type, n_ants, alpha, beta, rho, n_iterations, optimal_value):
        if instance_data is None:
            raise ValueError("No instance data provided")
            
        self.weight_type = weight_type
        if weight_type == 'EXPLICIT':
            self.distances = instance_data  # Ya tenemos la matriz de distancias
            self.n_cities = len(instance_data)
        else:
            self.cities = instance_data
            self.n_cities = len(instance_data)
            self.distances = self.calculate_distances()
            
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.n_iterations = n_iterations
        self.optimal_value = optimal_value
        
        # El resto de la inicialización se mantiene igual
        self.pheromone = np.ones((self.n_cities, self.n_cities))
        np.fill_diagonal(self.pheromone, 0)
        self.best_path = None
        self.best_distance = float('inf')
        self.convergence_iteration = None
        self.stabilization_count = 0
        self.prev_best = float('inf')

    def calculate_distances(self):
        """Calcula la matriz de distancias según el tipo de problema"""
        if self.weight_type == 'GEO':
            return self.calculate_geo_distances()
        elif self.weight_type == 'EUC_2D':
            return self.calculate_euc_distances()
        else:
            raise ValueError(f"Unsupported weight type: {self.weight_type}")
        
    def calculate_geo_distances(self):
        """
        Calcula distancias geográficas (GEO).
        
        La distancia geográfica se calcula usando la fórmula del gran círculo:
        dist = R * acos(0.5 * ((1 + cos(lng1 - lng2)) * cos(lat1 - lat2) - 
                            (1 - cos(lng1 - lng2)) * cos(lat1 + lat2))) + 1
        
        donde:
        - R es el radio de la Tierra (6378.388 km según TSPLIB)
        - lat1, lng1 son las coordenadas del primer punto en radianes
        - lat2, lng2 son las coordenadas del segundo punto en radianes
        
        Las coordenadas de entrada están en un formato especial donde:
        - Las coordenadas geográficas están codificadas como: GG.MM 
        - GG son los grados
        - MM son los minutos decimales
        
        Returns:
            numpy.ndarray: Matriz simétrica de distancias entre ciudades redondeadas al entero más cercano
        """
        distances = np.zeros((self.n_cities, self.n_cities))
        RRR = 6378.388  # Radio de la Tierra en kilómetros según TSPLIB
        
        def parse_degrees(coord):
            """
            Convierte una coordenada codificada en formato GG.MM a grados decimales.
            
            Fórmula: grados + (minutos * 5/3)
            donde minutos se obtiene como la parte decimal del número original
            """
            degrees = int(coord)
            minutes = coord - degrees
            return degrees + minutes * 5 / 3
        
        def to_radians(coord):
            """
            Convierte un par de coordenadas de grados a radianes.
            
            Fórmula: radianes = grados * π/180
            """
            lat, lng = coord
            lat_deg = parse_degrees(lat)
            lng_deg = parse_degrees(lng)
            return math.radians(lat_deg), math.radians(lng_deg)
        
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                # Convertir coordenadas a radianes
                lat1, lng1 = to_radians(self.cities[i])
                lat2, lng2 = to_radians(self.cities[j])
                
                # Calcular componentes de la fórmula del gran círculo
                q1 = math.cos(lng1 - lng2)
                q2 = math.cos(lat1 - lat2)
                q3 = math.cos(lat1 + lat2)
                distance = RRR * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1
                
                # Redondear al entero más cercano y almacenar en matriz simétrica
                dist = int(distance)
                distances[i][j] = dist
                distances[j][i] = dist
                
        return distances

    def calculate_euc_distances(self):
        """
        Calcula distancias euclidianas (EUC_2D).
        
        La distancia euclidiana entre dos puntos se calcula usando la fórmula:
        dist = √((x₁ - x₂)² + (y₁ - y₂)²)
        """
        distances = np.zeros((self.n_cities, self.n_cities))
        
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                # Calcular diferencias en coordenadas x e y
                dx = self.cities[i][0] - self.cities[j][0]
                dy = self.cities[i][1] - self.cities[j][1]
                
                # Calcular distancia euclidiana y redondear
                dist = int(math.sqrt(dx * dx + dy * dy) + 0.5)
                distances[i][j] = dist
                distances[j][i] = dist
                
        return distances
    
    def run(self):
        """Ejecuta el algoritmo principal"""
        start_time = time.time()
        
        for iteration in range(self.n_iterations):
            paths = []
            distances = []
            
            # Construcción de soluciones por cada hormiga
            for ant in range(self.n_ants):
                path = self.construct_solution()
                path_distance = self.calculate_path_distance(path)
                paths.append(path)
                distances.append(path_distance)
                
                # Actualizar mejor solución
                if path_distance < self.best_distance:
                    self.best_distance = path_distance
                    self.best_path = path.copy()
                    self.convergence_iteration = iteration
            
            # Actualizar feromonas
            self.update_pheromones(paths, distances)
            
            # Verificar estabilización
            if abs(self.prev_best - self.best_distance) < 1e-6:
                self.stabilization_count += 1
            else:
                self.stabilization_count = 0
            self.prev_best = self.best_distance
            
            # Detener si hay estabilización
            if self.stabilization_count >= 50:  # 50 iteraciones sin mejora
                break
        
        execution_time = time.time() - start_time
        gap = ((self.best_distance - self.optimal_value) / self.optimal_value) * 100
        
        return {
            'best_distance': self.best_distance,
            'gap': gap,
            'convergence_iteration': self.convergence_iteration,
            'execution_time': execution_time
        }
    
    def construct_solution(self):
        """Construye una solución para una hormiga"""
        unvisited = list(range(1, self.n_cities))
        path = [0]  # Comenzar desde la ciudad 0
        
        while unvisited:
            current = path[-1]
            probabilities = self.calculate_probabilities(current, unvisited)
            next_city = np.random.choice(unvisited, p=probabilities)
            path.append(next_city)
            unvisited.remove(next_city)
        
        return path
    
    def calculate_probabilities(self, current, unvisited):
        """Calcula probabilidades de selección para ciudades no visitadas"""
        if not unvisited:
            return np.array([])
            
        pheromone = np.array([self.pheromone[current][j] for j in unvisited])
        distance = np.array([1/max(1e-10, self.distances[current][j]) for j in unvisited])
        
        # Evitar división por cero
        probabilities = (pheromone ** self.alpha) * (distance ** self.beta)
        sum_prob = np.sum(probabilities)
        if sum_prob == 0:
            return np.ones(len(unvisited)) / len(unvisited)
            
        return probabilities / sum_prob
    
    def calculate_path_distance(self, path):
        """Calcula la distancia total de un camino"""
        total = 0
        for i in range(len(path)):
            total += self.distances[path[i]][path[(i+1) % self.n_cities]]
        return total
    
    def update_pheromones(self, paths, distances):
        """Actualiza los niveles de feromona"""
        # Evaporación
        self.pheromone *= (1 - self.rho)
        
        # Depósito
        for path, distance in zip(paths, distances):
            for i in range(len(path)):
                j = path[(i+1) % self.n_cities]
                self.pheromone[path[i]][j] += 1.0/distance
                self.pheromone[j][path[i]] += 1.0/distance  # Simétrico

def run_experiment():
    """Ejecuta el experimento completo"""
    # Parámetros experimentales
    params = {
        'n_ants': [10, 20, 30],
        'alpha': [0.5, 1.0, 1.5],
        'beta': [0.5, 1.0, 1.5],
        'rho': [0.1, 0.3, 0.5],
        'n_iterations': [1000, 2000]
    }
    
    # # Comentado por problemas de rendimiento (instancias muy grandes)
    # # Clasificar y seleccionar instancias aleatoriamente
    # classified = classify_tsps(TSPs)
    # selected_instances = {
    #     category: random.sample(instances, 3) 
    #     for category, instances in classified.items() 
    #     if instances
    # }

    # Instancias seleccionadas manualmente para el experimento
    selected_instances = {
        'pequeñas': ['burma14', 'ulysses16', 'gr17'],
        'medianas': ['eil51', 'berlin52', 'brazil58'],
        'grandes': ['eil101', 'lin105', 'pr107']
    }
    
    # Descargar todas las instancias
    print("\n[bold cyan]Iniciando fase de descarga de instancias...")
    available_instances = download_all_instances(selected_instances)
    
    if not available_instances:
        print("[bold red]No se pudo descargar ninguna instancia. Abortando experimento.")
        return
    
    print(f"\n[bold green]Instancias disponibles para procesamiento: {len(available_instances)}")
    print("[bold cyan]Iniciando fase de experimentación...")
    
    results = []
    
    # Procesar solo las instancias descargadas exitosamente
    for category, instances in selected_instances.items():
        print(f"\n[bold green]Procesando categoría: {category}")
        
        for instance_name in instances:
            if instance_name not in available_instances:
                print(f"[yellow]Saltando instancia {instance_name} (no disponible)")
                continue
                
            print(f"\n[blue]Procesando instancia: {instance_name}")
            
            # Cargar instancia
            instance_data, weight_type = load_tsp_instance(instance_name)
            if instance_data is None:
                print(f"[red]Error cargando instancia {instance_name}, saltando...")
                continue
                
            optimal_value = TSPs[instance_name] # Valor óptimo conocido
            
            # Crea todas las combinaciones únicas de los parámetros del experimento (162 combinaciones)
            param_combinations = list(product(
                params['n_ants'],
                params['alpha'],
                params['beta'],
                params['rho'],
                params['n_iterations']
            ))
            
            for n_ants, alpha, beta, rho, n_iter in tqdm(param_combinations, 
                                                        desc=f"Procesando {instance_name}"):
                try:
                    # Inicializa el ACO con los parámetros seleccionados
                    aco = AntColonyOptimizer(
                        instance_data=instance_data,
                        weight_type=weight_type,
                        n_ants=n_ants,
                        alpha=alpha,
                        beta=beta,
                        rho=rho,
                        n_iterations=n_iter,
                        optimal_value=optimal_value
                    )
                    
                    # Ejecuta el algoritmo
                    result = aco.run()
                    
                    # Guarda los resultados
                    results.append({
                        'instance': instance_name,
                        'category': category,
                        'n_ants': n_ants,
                        'alpha': alpha,
                        'beta': beta,
                        'rho': rho,
                        'n_iterations': n_iter,
                        'best_distance': result['best_distance'],
                        'gap': result['gap'],
                        'convergence_iteration': result['convergence_iteration'],
                        'execution_time': result['execution_time']
                    })
                except Exception as e:
                    print(f"[red]Error ejecutando ACO para {instance_name}: {str(e)}")
                    continue
    
    if not results:
        print("[bold red]No se obtuvieron resultados. El experimento no pudo completarse.")
        return
        
    # Convertir resultados a DataFrame y guardar
    df_results = pd.DataFrame(results)
    df_results.to_csv('experiment_results.csv', index=False)
    
    # Análisis de resultados
    analyze_results(df_results)

if __name__ == "__main__":
    run_experiment()