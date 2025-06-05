import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
##########################################################
def getFaceVertices(face):
    vertices = []
    start_halfEdge = face.halfEdge
    current_halfEdge = start_halfEdge
    while True:
        vertices.append(current_halfEdge.vertex)
        current_halfEdge = current_halfEdge.next
        if current_halfEdge == start_halfEdge:
            break
    return vertices

# NUEVO: función escalar f: usamos coordenada z
def scalar_function(vertex):
    return vertex.z  # Puedes cambiar esto por otras funciones si deseas

# NUEVO: agrupar caras por niveles de la función f
def group_faces_by_scalar_level(facesArray, num_levels=10):
    face_levels = {}
    f_values = {}

    # Obtener valor f promedio por cara
    for face in facesArray.values():
        vertices = getFaceVertices(face)
        avg_f = sum(scalar_function(v) for v in vertices) / len(vertices)
        f_values[face] = avg_f

    # Crear niveles
    levels = create_scalar_levels(f_values, num_levels)

    # Asignar cada cara a un nivel
    for face, avg_f in f_values.items():
        for level_idx, (low, high) in enumerate(levels):
            if low <= avg_f < high or (level_idx == num_levels - 1 and avg_f == high):
                if level_idx not in face_levels:
                    face_levels[level_idx] = []
                face_levels[level_idx].append(face)
                break

    return face_levels

# NUEVO: conteo de componentes conexas por nivel
def compute_connected_components_by_level(facesArray, num_levels=10):
    levels = group_faces_by_scalar_level(facesArray, num_levels)
    result = {}

    for level, faces_in_level in levels.items():
        sub_faces_dict = {i: f for i, f in enumerate(faces_in_level)}

        # Usamos la misma función de componentes conexas con subgrupo de caras
        visited = {}
        components = 0

        for face in sub_faces_dict.values():
            if not visited.get(face, False):
                components += 1
                dfs_stack = [face]

                while dfs_stack:
                    current = dfs_stack.pop()
                    if visited.get(current, False):
                        continue
                    visited[current] = True

                    start = current.halfEdge
                    edge = start
                    while True:
                        if edge.twin:
                            neighbor = edge.twin.face
                            if neighbor in sub_faces_dict.values() and not visited.get(neighbor, False):
                                dfs_stack.append(neighbor)
                        edge = edge.next
                        if edge == start:
                            break
        result[level] = components
    return result

def create_scalar_levels(f_values, num_levels=10):
    """
    Divide los valores escalares en niveles equiespaciados.
    f_values: diccionario {face: valor_f}
    Devuelve una lista de tuplas [(min1, max1), (min2, max2), ...]
    """
    if not f_values:
        return []

    min_f = min(f_values.values())
    max_f = max(f_values.values())
    step = (max_f - min_f) / num_levels if max_f > min_f else 1e-6

    levels = []
    for i in range(num_levels):
        lower = min_f + i * step
        upper = min_f + (i + 1) * step
        levels.append((lower, upper))
    return levels

def get_vertices_in_level(vertices_dict, lower_bound, upper_bound, scalar_func=scalar_function):
    """
    Devuelve una lista de vértices cuyo valor escalar cae dentro del intervalo [lower_bound, upper_bound).
    
    Parámetros:
        vertices_dict: diccionario {índice: Vertex}
        lower_bound: límite inferior del intervalo
        upper_bound: límite superior del intervalo
        scalar_func: función escalar que toma un Vertex y devuelve un valor numérico (por defecto: z)
    
    Retorna:
        Lista de vértices en ese intervalo.
    """
    result = []
    for vertex in vertices_dict.values():
        value = scalar_func(vertex)
        if lower_bound <= value < upper_bound:
            result.append(vertex)
    return result

def get_vertex_neighbors(vertex):
    """
    Devuelve una lista de vértices adyacentes (conectados por una arista) usando la estructura Half-Edge.
    """
    neighbors = []
    start = vertex.halfEdge
    if start is None:
        return neighbors  # vértice aislado

    edge = start
    visited_edges = set()

    while True:
        if edge.next:
            next_vertex = edge.next.origin
            if next_vertex not in neighbors:
                neighbors.append(next_vertex)
        visited_edges.add(edge)

        if edge.twin is None or edge.twin.next is None:
            break
        edge = edge.twin.next

        if edge == start or edge in visited_edges:
            break
    return neighbors

def get_vertex_neighbors(vertex):
    """
    Devuelve una lista de vértices adyacentes (conectados por una arista) usando la estructura Half-Edge.
    """
    neighbors = []
    start = vertex.halfEdge
    if start is None:
        return neighbors  # vértice aislado

    edge = start
    visited_edges = set()

    while True:
        if edge.next:
            next_vertex = edge.next.origin
            if next_vertex not in neighbors:
                neighbors.append(next_vertex)
        visited_edges.add(edge)

        if edge.twin is None or edge.twin.next is None:
            break
        edge = edge.twin.next

        if edge == start or edge in visited_edges:
            break
    return neighbors

def find_connected_vertex_components(vertices_subset):
    """
    Encuentra componentes conexas en un conjunto de vértices usando DFS y Half-Edge.
    
    Parámetro:
        vertices_subset: lista de vértices (todos con f(v) ∈ [a, b))
    
    Retorna:
        Lista de listas, donde cada sublista es una componente conexa.
    """
    visited = set()
    components = []

    vertex_set = set(vertices_subset)

    for vertex in vertices_subset:
        if vertex in visited:
            continue

        # Comenzamos una nueva componente
        stack = [vertex]
        current_component = []

        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            current_component.append(v)

            for neighbor in get_vertex_neighbors(v):
                if neighbor in vertex_set and neighbor not in visited:
                    stack.append(neighbor)

        components.append(current_component)
    return components

def build_reeb_graph(vertices_dict, num_levels=10):
    """
    Construye un grafo de Reeb a partir de los vértices divididos por niveles de la función escalar.
    
    Retorna:
        G: grafo de NetworkX
        node_mapping: diccionario {nodo_id: [lista de vértices]}
    """
    f_values = {v: scalar_function(v) for v in vertices_dict.values()}
    levels = create_scalar_levels(f_values, num_levels)
    G = nx.Graph()
    node_mapping = {}  # nodo_id: lista de vértices
    level_components = []

    node_id = 0
    for level_idx, (low, high) in enumerate(levels):
        verts_in_level = get_vertices_in_level(vertices_dict, low, high)
        components = find_connected_vertex_components(verts_in_level)
        level_nodes = []

        for comp in components:
            G.add_node(node_id, level=level_idx, size=len(comp))
            node_mapping[node_id] = set(comp)
            level_nodes.append(node_id)
            node_id += 1

        level_components.append(level_nodes)

    # Conectar nodos entre niveles consecutivos si comparten algún vértice
    for l in range(len(level_components) - 1):
        for u in level_components[l]:
            for v in level_components[l + 1]:
                if node_mapping[u] & node_mapping[v]:
                    G.add_edge(u, v)

    return G, node_mapping

import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_reeb_graph(G):
    """
    Dibuja el grafo de Reeb con nodos coloreados por nivel y solo número como etiqueta.
    """
    pos = {}
    levels = nx.get_node_attributes(G, 'level')

    # Colores por nivel
    unique_levels = sorted(set(levels.values()))
    cmap = cm.get_cmap('tab10', len(unique_levels))
    level_to_color = {lvl: mcolors.to_hex(cmap(i)) for i, lvl in enumerate(unique_levels)}
    node_colors = [level_to_color[levels[n]] for n in G.nodes]

    # Posiciones simples: x = nivel, y = -nodo_id (para ordenarlos de arriba a abajo)
    for node in G.nodes:
        pos[node] = (levels[node], -node)

    plt.figure(figsize=(10, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels={n: str(n) for n in G.nodes},
        node_color=node_colors,
        node_size=800,
        font_size=10,
        font_weight='bold',
        edge_color='gray'
    )
    plt.title("Grafo de Reeb (colores por nivel)")
    plt.xlabel("Nivel escalar (f)")
    plt.axis('off')
    plt.show()

