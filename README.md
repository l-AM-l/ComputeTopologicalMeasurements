# Compute Topological Measurements (borrador lo que creo que va el proyecto)

This project analyzes the topological structure of a 3D mesh using the **Half-Edge Data Structure (HEDS)**. It was developed as a final project for a computational geometry course.

---

## Project Goals

The goal of this project is to:
- Convert a mesh from `.obj` format into the Half-Edge Data Structure (HEDS).
- Detect **connected components** in the mesh.
- Analyze the **spatial relationship** between components (e.g., whether one is inside another).
- Identify and count **open boundaries** (holes).
- Compute the **genus** (number of "handles" or "holes").
- (Optional) Correct the orientation of the normals if necessary.

---

## Files

- `mesh_processin_03.py`: Main script to load the mesh, build HEDS, analyze the mesh, and output results.
- `reeb.py`: script to create reeb graph, all fuctions realated are there.
- `input.obj`: Example input mesh.
- `output.obj`: Output mesh with corrected orientation (if needed).

---

## libraries

To run the project it is necessary to install the following libraries, it can be done from terminal using `pip install`

- `numpy`
- `matplotlib`
- `networkx`

---

## Team

- Ana María Guzmán Solís (0252231)
- Raquel Magdalena Ochoa Martínez (0235324)

---
