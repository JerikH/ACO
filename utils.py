import matplotlib.pyplot as plt
import os
import tsplib95
import urllib.request
import gzip
import numpy as np
from rich import print

# Diccionario con las instancias TSP y sus soluciones óptimas
TSPs = {"att48":10628,"att532":27686,"a280":2579,"ali535":202339,"bayg29":1610,"bays29":2020,"bier127":118282,"brazil58":25395,"brd14051":469385,"berlin52":7542,"burma14":3323,"brg180":1950,"ch130":6110,"ch150":6528,"d198":15780,"d1291":50801,"d657":48912,"d2103":80450,"d493":35002,"d1655":62128,"d15112":1573084,"dantzig42":699,"d18512":645238,"eil51":426,"dsj1000":18659688,"eil76":538,"fl417":11861,"fl1577":22249,"fl1400":20127,"eil101":629,"fl3795":28772,"fnl4461":182566,"fri26":937,"gil262":2378,"gr17":2085,"gr120":6942,"gr48":5046,"gr21":2707,"gr96":55209,"gr24":1272,"gr137":69853,"gr229":134602,"gr202":40160,"gr431":171414,"gr666":294358,"kroA100":21282,"kroB100":22141,"hk48":11461,"kroD100":21294,"kroC100":20749,"kroE100":22068,"kroA150":26524,"kroB150":26130,"kroA200":29368,"kroB200":29437,"lin105":14379,"nrw1379":56638,"p654":34643,"lin318":42029,"linhp318":41345,"pa561":2763,"pcb442":50778,"pcb1173":56892,"pcb3038":137694,"pla7397":23260728,"pla85900":142382641,"pla33810":66048945,"pr76":108159,"pr107":44303,"pr124":59030,"pr136":96772,"pr144":58537,"pr226":80369,"pr152":73682,"pr264":49135,"pr299":48191,"pr1002":259045,"pr2392":378032,"pr439":107217,"rat99":1211,"rat195":2323,"rat575":6773,"rat783":8806,"rd100":7910,"rd400":15281,"rl1323":270199,"rl5915":565530,"rl1889":316536,"rl1304":252948,"rl5934":556045,"rl11849":923288,"si175":21407,"si535":48450,"si1032":92650,"st70":675,"ts225":126643,"u159":42080,"tsp225":3916,"swiss42":1273,"u574":36905,"u1060":224094,"u724":41910,"u1432":152970,"u1817":57201,"u2152":64253,"usa13509":19982859,"u2319":234256,"ulysses22":7013,"ulysses16":6859,"vm1084":239297,"vm1748":336556}

def plot_best_path(instance_name, coords, path):
    """
    Visualiza el mejor camino encontrado para una instancia TSP.
    
    Args:
        instance_name (str): Nombre de la instancia
        coords (np.ndarray): Coordenadas de las ciudades
        path (list): Mejor camino encontrado
    """
    plt.figure(figsize=(10, 10))
    
    # Plotear todas las ciudades
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=50)
    
    # Plotear el camino
    path_coords = np.array([coords[i] for i in path + [path[0]]])
    plt.plot(path_coords[:, 0], path_coords[:, 1], 'b-', linewidth=1, alpha=0.7)
    
    # Añadir etiquetas a las ciudades
    for i, (x, y) in enumerate(coords):
        plt.annotate(f'City {i}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.title(f'Best Path for {instance_name}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # Crear directorio para las visualizaciones si no existe
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(f'visualizations/{instance_name}_best_path.png')
    plt.close()

def analyze_results(df):
    """Analiza y visualiza los resultados del experimento"""
    print("\n[bold blue]===================== ANÁLISIS DETALLADO DE RESULTADOS =====================")
    print("\n[bold cyan]Mejores soluciones por instancia:")
    print(f"{'Instancia':<10} {'Categoría':<10} {'Distancia':<10} {'GAP(%)':<8} {'Tiempo(s)':<10} {'n_ants':<6} {'alpha':<6} {'beta':<6} {'rho':<5} {'iter':<6} {'conv':<6}")
    print("-" * 100)
    
    # Análisis por instancia
    for instance in df['instance'].unique():
        instance_data = df[df['instance'] == instance]
        
        # Encontrar la mejor solución (menor GAP y tiempo de ejecución)
        best_solutions = instance_data[instance_data['gap'] == instance_data['gap'].min()]
        best_solution = best_solutions.loc[best_solutions['execution_time'].idxmin()]
        
        # Cargar las coordenadas de la instancia para la visualización
        instance_data, weight_type = load_tsp_instance(instance)
        if instance_data is not None and weight_type in ['EUC_2D', 'GEO']:
            try:
                plot_best_path(instance, instance_data, best_solution['best_path'])
                print(f"[green]Visualización guardada para {instance}")
            except Exception as e:
                print(f"[red]Error generando visualización para {instance}: {str(e)}")
        
        print(f"{best_solution['instance']:<10} "
              f"{best_solution['category']:<10} "
              f"{best_solution['best_distance']:<10.1f} "
              f"{best_solution['gap']:<8.2f} "
              f"{best_solution['execution_time']:<10.4f} "
              f"{best_solution['n_ants']:<6} "
              f"{best_solution['alpha']:<6.1f} "
              f"{best_solution['beta']:<6.1f} "
              f"{best_solution['rho']:<5.1f} "
              f"{best_solution['n_iterations']:<6} "
              f"{best_solution['convergence_iteration']:<6}")
    
    # Análisis global por categoría
    print("\n[bold blue]===================== ANÁLISIS GLOBAL =====================")
    print("\n[bold cyan]Resumen por categoría:")
    print(f"{'Categoría':<10} {'GAP(%)':<16} {'Tiempo(s)':<16}")
    print(f"{'':10} {'prom':>8} {'min':>7} {'prom':>8} {'min':>7}")
    print("-" * 50)
    
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        print(f"{category:<10} "
              f"{cat_data['gap'].mean():>8.2f} "
              f"{cat_data['gap'].min():>7.2f} "
              f"{cat_data['execution_time'].mean():>8.2f} "
              f"{cat_data['execution_time'].min():>7.2f}")
    
    # Promedio de mejores parámetros por categoría
    print("\n[bold cyan]Promedio de mejores parámetros por categoría:")
    print(f"{'Categoría':<10} {'n_ants':>8} {'alpha':>8} {'beta':>8} {'rho':>8}")
    print("-" * 50)
    
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        best_solutions = cat_data.loc[cat_data.groupby('instance')['gap'].idxmin()]
        means = best_solutions[['n_ants', 'alpha', 'beta', 'rho']].mean()
        
        print(f"{category:<10} "
              f"{means['n_ants']:>8.1f} "
              f"{means['alpha']:>8.1f} "
              f"{means['beta']:>8.1f} "
              f"{means['rho']:>8.1f}")

def classify_tsps(tsps):
    """Clasifica las instancias TSP por tamaño"""
    result = {'pequeñas': [], 'medianas': [], 'grandes': []}
    
    for name in tsps.keys():
        cities = int(''.join(filter(str.isdigit, name)))
        
        if 10 <= cities <= 20:
            result['pequeñas'].append(name)
        elif 50 <= cities <= 100:
            result['medianas'].append(name)
        elif cities > 100:
            result['grandes'].append(name)
            
    return {k: sorted(v) for k, v in result.items()}

def download_all_instances(selected_instances):
    """Descarga todas las instancias seleccionadas antes de comenzar el experimento"""
    print("[bold blue]Fase de descarga de instancias:")
    
    # Crear directorio si no existe
    os.makedirs("instances", exist_ok=True)
    
    # Mantener registro de instancias descargadas exitosamente
    successfully_downloaded = []
    
    for category, instances in selected_instances.items():
        print(f"\n[bold green]Descargando instancias de categoría: {category}")
        
        for instance_name in instances:
            # Verificar si ya existe el archivo
            if os.path.exists(f"instances/{instance_name}.tsp"):
                print(f"[yellow]Instancia {instance_name} ya existe localmente, omitiendo descarga...")
                successfully_downloaded.append(instance_name)
                continue
                
            url = f"http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/{instance_name}.tsp.gz"
            print(f"[blue]Descargando {instance_name}...")
            
            try:
                # Descargar y descomprimir el archivo
                response = urllib.request.urlopen(url)
                with gzip.open(response, 'rb') as f_in:
                    with open(f"instances/{instance_name}.tsp", 'wb') as f_out:
                        f_out.write(f_in.read())
                print(f"[green]✓ {instance_name} descargado exitosamente")
                successfully_downloaded.append(instance_name)
            except Exception as e:
                print(f"[red]Error descargando {instance_name}: {str(e)}")
    
    return successfully_downloaded

def load_tsp_instance(instance_name):
    """Carga una instancia TSP usando tsplib95 y retorna las coordenadas o matriz de distancias"""
    try:
        # Cargar el problema
        problem = tsplib95.load(f"instances/{instance_name}.tsp")
        
        # Verificar el tipo de problema
        weight_type = problem.edge_weight_type # 'EXPLICIT', 'EUC_2D', 'GEO'
        
        if weight_type == 'EXPLICIT':
            # Para problemas con matriz de distancias en lugar de coordenadas.
            n = problem.dimension
            distances = np.zeros((n, n))
            
            # Obtener las distancias directamente del problema
            for i, j in problem.get_edges():
                distances[i-1][j-1] = problem.get_weight(i, j)
                distances[j-1][i-1] = problem.get_weight(i, j)  # matriz simétrica
                
            return distances, 'EXPLICIT'
            
        elif weight_type in ['EUC_2D', 'GEO']:
            # Para problemas basados en coordenadas
            nodes = list(problem.get_nodes())
            coords = []
            for node in nodes:
                x, y = problem.get_display(node)
                coords.append([float(x), float(y)])
            return np.array(coords), weight_type
            
        else:
            print(f"Tipo de problema no soportado: {weight_type}")
            return None, None
            
    except Exception as e:
        print(f"Error cargando {instance_name}: {str(e)}")
        return None, None
