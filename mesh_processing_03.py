
# Computational Geometry final project 2025
import numpy as np
import matplotlib.pyplot as plt



# read obj file
def readObjFile(file_path):
    vertices = []
    faces = []
    
    with open(file_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):
                #mete las palabras que inicien con "v" a la lista sindo esta los vertices
                vertex = list(map(float, line.strip().split()[1:]))
                vertices.append(vertex)
            elif line.startswith('f '):
                 #mete las palabras que inicien con "f" a la lista sindo esta los faces
                face = list(map(int, line.strip().split()[1:]))
                face = [index - 1 for index in face]
                faces.append(face)
    return vertices, faces

# plot the mesh (optional, we will use meshlab) 

# convert to HEDS (half-edges data structure)
class Vertex:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.halfEdge = None

class Face:
    def __init__(self):
        self.halfEdge = None

class HalfEdge:
    def __init__(self):
        self.origin = None
        self.twin = None
        self.face = None
        self.next = None
        self.prev = None


def VFtoHEDS(vertices, faces):
    # create vertices
    verticesArray = {}
    for i, (x, y, z) in enumerate(vertices):
        vertex = Vertex(x, y, z)
        verticesArray[i] = vertex

    # Create half edges
    halfEdgesArray = [HalfEdge() for _ in range(len(faces)*3)]

    # Create faces
    facesArray = {}
    for i, _ in enumerate(faces):
        face = Face()
        facesArray[i] = face

    # Connect halfEdges to vertices
    for face_index, face_vertices in enumerate(faces): 
        for i, vertex_index in enumerate(face_vertices):
            halfEdge = halfEdgesArray[face_index *3 +i]
            halfEdge.vertex = verticesArray[vertex_index]
            verticesArray[vertex_index].halfEdge = halfEdge
    
    # Connect half edges to faces
    for face_index, face_vertices in enumerate(faces):
        face = facesArray[face_index]
        face_halfEdgesArray = [halfEdgesArray[face_index * 3 + i] for i in range(3)]
        face.halfEdge = face_halfEdgesArray[0]
        for i in range(3):
            face_halfEdgesArray[i].face = face
            face_halfEdgesArray[i].next = face_halfEdgesArray[(i + 1) % 3]
            face_halfEdgesArray[i].prev = face_halfEdgesArray[(i + 2) % 3]
    
    #to change memory dorection to actual numbers
    vertex_to_index = {v: idx for idx, v in verticesArray.items()}

    # Connect twins
    #######################################################################################
    edge_dict = {} #dictionary to track edges to mach with their twins
    for face_index, face_vertices in enumerate(faces):
        print(f"\nCara {face_index}: Vertices {face_vertices}")
        for i in range(3):
            curr = face_vertices[i] #current vertex
            next_v = face_vertices[(i + 1) % 3]  #next vertex (the twin)
            he = halfEdgesArray[face_index * 3 + i] #current half edge 
            
            he.origin = verticesArray[curr]  #start point
            he.next = halfEdgesArray[face_index * 3 + (i + 1) % 3]#next half edge in the face
            
            key = (min(curr, next_v), max(curr, next_v))
            if key in edge_dict:
                twin = edge_dict[key] #twin has been found 
                he.twin = twin#link current hald edge to twin 
                twin.twin = he#link twin bak to current
                print(f"conectado twon: {curr}→{next_v} <-> {vertex_to_index[twin.origin]}→{vertex_to_index[twin.next.origin]}")
            else:
                edge_dict[key] = he #store for later 
                print(f"Añadido a diccionario (sin twin aún)")

    # Debuggin for finding all the twins
    print("\n=== verificacion de twins ===")
    for he in halfEdgesArray:
        if he.twin:
            # use vertex_to_index so that we can use numbers instead of the memory
            origin_idx = vertex_to_index[he.origin]
            next_idx = vertex_to_index[he.next.origin]
            twin_origin_idx = vertex_to_index[he.twin.origin]
            twin_next_idx = vertex_to_index[he.twin.next.origin]
            print(f"Half-Edge {origin_idx}→{next_idx} | Twin: {twin_origin_idx}→{twin_next_idx}")
        else:
            # the borders
            origin_idx = vertex_to_index[he.origin]
            next_idx = vertex_to_index[he.next.origin]
            print(f"Half-Edge {origin_idx}→{next_idx} | Twin: no(borde)")
       #########################################################################################3 
    return verticesArray, halfEdgesArray, facesArray

# convert back to VF (Vertices Faces data structure)
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

def HEDStoVF(verticesArray, halfEdgesArray, facesArray):
    vertices = [(vertex.x, vertex.y, vertex.z) for vertex in verticesArray.values()]
    faces = []
    for face in facesArray.values():
        face_vertices = getFaceVertices(face)
        face_indices = [list(verticesArray.keys())[list(verticesArray.values()).index(vertex)] for vertex in face_vertices]
        faces.append(face_indices)
    return vertices, faces

# write the final .obj or .ply file (with color coding)
def writeObjFile(vertices, faces, output_file):
    with open(output_file, 'w') as obj_file:
        for vertex in vertices:
            obj_file.write('v ' + ' '.join(map(str, vertex)) + '\n')
        for face in faces:
            obj_file.write('f ' + ' '.join(map(lambda x: str(x + 1), face)) + '\n')

def visualizeMesh(vertices, faces, vertex_color='k', edge_color='b'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot vertices
    vertices_array = np.array(vertices)
    ax.scatter(vertices_array[:,0], vertices_array[:,2], vertices_array[:,1], c=vertex_color, depthshade=False)
    
    # Plot edges
    for face in faces:
        face_vertices = [vertices[i] for i in face]
        face_vertices.append(vertices[face[0]])  # Close the loop
        face_vertices = np.array(face_vertices)
        ax.plot(face_vertices[:,0], face_vertices[:,2], face_vertices[:,1], c=edge_color)  
        
    # Set equal aspect ratio
    ax.set_box_aspect([np.ptp(vertices_array[:,0]), np.ptp(vertices_array[:,0]), np.ptp(vertices_array[:,1])])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()


# read obj
vertices, faces = readObjFile('input.obj')

# visualize the mesh
visualizeMesh(vertices, faces)   

# convert to HEDS
verticesArray, halfEdgesArray, facesArray = VFtoHEDS(vertices, faces)

# Any mesh processing operation (your final project) using HEDS structure

# convert back to VF
newVertices, newFaces = HEDStoVF(verticesArray, halfEdgesArray, facesArray)

# write the final result
writeObjFile(newVertices, newFaces, 'output.obj')