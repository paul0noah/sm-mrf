from pathlib import Path
import os
import igl
import polyscope as ps
ps.init()
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import pymeshfix as pymeshfix
import numpy as np


def knn_search(x, X, k=1):
    """
    find indices of k-nearest neighbors of x in X
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    _, indices = nbrs.kneighbors(x)
    if k == 1:
        return indices.flatten()
    else:
        return indices


def read_shape(file, as_cloud=False):
    """
    Read mesh from file.

    Args:
        file (str): file name
        as_cloud (bool, optional): read shape as point cloud. Default False
    Returns:
        verts (np.ndarray): vertices [V, 3]
        faces (np.ndarray): faces [F, 3] or None
    """
    if as_cloud:
        verts = np.asarray(o3d.io.read_point_cloud(file).points)
        faces = None
    else:
        mesh = o3d.io.read_triangle_mesh(file)
        verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)

    return verts, faces

def merge_from_array(v, f):
    """
    Create topological noisy shape by merging self intersections.
    This might substantially change the triangulation
    Args:
        v: numpy array with |v|x3 vertex coordinates
        f: numpy array with |f|x3 indices into v (each row forming a triangle)

    Returns:
        v_merged: numpy array with |v|x3 vertex coordinates
        f_merged: numpy array with |f|x3 indices into v (each row forming a triangle)
        idx_v_merged_into_v: numpy array with |v|x1 of vertices from v_merged into v
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)

    pc = o3d.geometry.PointCloud()
    normals = igl.per_vertex_normals(v, f)

    # make sure that surfaces touch each other in a reproducable way (would use random otherwise)
    v += 0.0008 * normals

    pc.points = o3d.utility.Vector3dVector(v)

    pc.normals = o3d.utility.Vector3dVector(normals)

    # from open3d example
    # https://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html#Poisson-surface-reconstruction
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=9)

    v_recon, f_recon = np.asarray(mesh.vertices),  np.asarray(mesh.triangles)
    #ps.register_surface_mesh("xre", v_recon, f_recon)
    v_recon, f_recon = pymeshfix.clean_from_arrays(v_recon, f_recon, joincomp=True)
    #ps.register_surface_mesh("xclean", v_recon, f_recon)
    #ps.show()

    _, v_recon, f_recon, _, _ = igl.qslim(v_recon, f_recon, f.shape[0])
    idx = knn_search(v_recon, v)
    idx_ = knn_search(v, v_recon)
    return v_recon, f_recon, idx, idx_


dataset_dir = "datasets/FAUST_r/"
output_dir = "datasets/TOPFAUST_r/"

files = "off/"
corres = "corres/"

data_dir_files = os.path.join(dataset_dir, files)
data_dir_corres = os.path.join(dataset_dir, corres)
output_dir_files = os.path.join(output_dir, files)
output_dir_corres = os.path.join(output_dir, corres)

# make dirs
if not os.path.exists(output_dir_files):
    path = Path(output_dir_files)
    path.mkdir(parents=True)
if not os.path.exists(output_dir_corres):
    path = Path(output_dir_corres)
    path.mkdir(parents=True)


files = os.listdir(data_dir_files)
files = [i for i in files if i.endswith('.off')]
files = sorted(files)

import polyscope as ps
ps.init()

for file in files:
    inst_name = file.replace(".off", "")
    print(inst_name)
    shape_file = os.path.join(data_dir_files, file)
    output_shape_file = os.path.join(output_dir_files, file)
    if os.path.exists(output_shape_file):
        print("-> exists")
        continue

    v, f = read_shape(shape_file)
    igl.remove_unreferenced(v, f)

    try:
        vm, fm, idxm, idx_m = merge_from_array(v, f)
    except:
        continue

    ps.register_surface_mesh("orig", v, f)
    ps.register_surface_mesh("remeshed", vm, fm)
    #ps.show()

    corres_file = os.path.join(data_dir_corres, inst_name + ".vts")
    corr = np.loadtxt(corres_file, dtype=np.int32) - 1

    corrm = idx_m[corr]

    c = np.arange(corr.shape[0])

    p = ps.register_point_cloud("corr", v[corr])
    p.add_scalar_quantity(".", c, enabled=True)

    p = ps.register_point_cloud("corr_", vm[corrm])
    p.add_scalar_quantity(".", c, enabled=True)



    igl.write_triangle_mesh(output_shape_file, vm, fm)

    output_corres_file = os.path.join(output_dir_corres, inst_name + ".vts")
    corrm += 1
    np.savetxt(output_corres_file, corrm)
    #ps.show()
