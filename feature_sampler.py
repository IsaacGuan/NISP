import os
import igl
import trimesh
import numpy as np

from PIL import Image
from mesh_to_sdf import sample_sdf_near_surface
from decimal import *
from utils import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SHAPES_DIR = os.path.join(DATA_DIR, 'shapes')
DIFFUSE_MAPS_DIR = os.path.join(DATA_DIR, 'diffuse-maps')

# modified from https://github.com/daviesthomas/overfitSDF/blob/main/neuralImplicit/geometry.py#L55
class UniformSampler():
    def __init__(self, mesh, ratio=0.0, std=0.0, vertice_sampling=False):
        self.V = mesh.vertices
        self.F = mesh.faces
        self.sample_vertices = vertice_sampling

        if ratio < 0 or ratio > 1:
            raise('Ratio must be [0,1]')

        self.ratio = ratio

        if std < 0 or std > 1:
            raise('Normal deviation must be [0,1]')

        self.std = std

        self.calculate_face_bins()

    def calculate_face_bins(self):
        vc = np.cross(
            self.V[self.F[:, 0], :] - self.V[self.F[:, 2], :],
            self.V[self.F[:, 1], :] - self.V[self.F[:, 2], :])

        A = np.sqrt(np.sum(vc ** 2, 1))
        FA = A / np.sum(A)
        self.face_bins = np.concatenate(([0],np.cumsum(FA))) 

    def surface_samples(self, n):
        R = np.random.rand(n)
        sample_face_idxs = np.array(np.digitize(R,self.face_bins)) -1

        r = np.random.rand(n, 2)
        A = self.V[self.F[sample_face_idxs, 0], :]
        B = self.V[self.F[sample_face_idxs, 1], :]
        C = self.V[self.F[sample_face_idxs, 2], :]
        P = (1 - np.sqrt(r[:,0:1])) * A \
                + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B \
                + np.sqrt(r[:,0:1]) * r[:,1:] * C

        return P

    def vertice_samples(self, n):
        verts = np.random.choice(len(self.V), n)
        return self.V[verts]
    
    def normal_dist(self, V):
        if self.std > 0.0:
            return np.random.normal(loc=V, scale=self.std)

        return V
        
    def random_samples(self, n):
        points = np.array([])
        while points.shape[0] < n:
            remaining_points = n - points.shape[0]
            p = (np.random.rand(remaining_points, 3) - 0.5)*2
            # p = p[np.linalg.norm(p, axis=1) <= SAMPLE_SPHERE_RADIUS]

            if points.size == 0:
                points = p 
            else:
                points = np.concatenate((points, p))
        return points

    def sample(self, n):
        n_random = round(Decimal(n)*Decimal(self.ratio))
        n_surface = n - n_random

        x_random = self.random_samples(n_random)

        if n_surface > 0:
            if self.sample_vertices:
                x_surface = self.vertice_samples(n_surface)
            else:
                x_surface = self.surface_samples(n_surface)

            x_surface = self.normal_dist(x_surface)
            if n_random > 0:
                x = np.concatenate((x_surface, x_random))
            else:
                x = x_surface
        else:
            x = x_random

        np.random.shuffle(x)

        return x

# modified from https://github.com/daviesthomas/overfitSDF/blob/main/neuralImplicit/geometry.py#L155
class ImportanceSampler():
    def __init__(self, mesh, M, W):
        self.mesh = mesh
        self.M = M
        self.W = W
    
        self.uniform_sampler = UniformSampler(self.mesh, ratio=1.0)

    def subsample(self, s, N):
        w = np.exp(-self.W*np.abs(s))
        pU = w / np.sum(w)
        C = np.concatenate(([0],np.cumsum(pU)))
        C = C[0:-1]

        R = np.random.rand(N)

        I = np.array(np.digitize(R,C)) - 1

        return I

    def sample(self, N):
        U = self.uniform_sampler.sample(self.M)
        s, _, _ = igl.signed_distance(U, self.mesh.vertices, self.mesh.faces)
        I = self.subsample(s, N)

        R = np.random.choice(len(U), int(N*0.1))
        S = U[I,:] # np.concatenate((U[I,:],U[R,:]), axis=0)
        return S

class FeatureSampler:
    def __init__(self):
        self.model_name = None
        self.mesh = None
        self.vertices = None
        self.vertices_tex = None
        self.vertices_norm = None
        self.faces = None
        self.faces_tex = None
        self.faces_norm = None
        self.tex = None
        self.point_samples = None
        self.surface_points = None
        self.component_samples = None
        self.distance_samples = None
        self.uv_samples = None
        self.color_samples = None

    def load_object(self, model_name):
        self.model_name = model_name
        model_file = os.path.join(SHAPES_DIR, self.model_name + '.obj')

        if os.path.exists(model_file):
            self.mesh = trimesh.load(model_file, process=False, maintain_order=True)

            f = open(model_file, 'r')
            text = f.read()
            text = trimesh.util.decode_text(text)
            text = '\n{}\n'.format(text.strip().replace('\r\n', '\n'))
            text = text.replace('\\\n', '')

            self.vertices, self.vertices_norm, self.vertices_tex, vc = trimesh.exchange.obj._parse_vertices(text=text)

            face_tuples = trimesh.exchange.obj._preprocess_faces(text=text)
            material, current_object, chunk = face_tuples.pop()
            face_lines = [i.split('\n', 1)[0] for i in chunk.split('\nf ')[1:]]
            joined = ' '.join(face_lines).replace('/', ' ')
            array = np.fromstring(joined, sep=' ', dtype=np.int64) - 1
            columns = len(face_lines[0].strip().replace('/', ' ').split())
            self.faces, self.faces_tex, self.faces_norm = trimesh.exchange.obj._parse_faces_vectorized(
                array=array, columns=columns, sample_line=face_lines[0])

        tex_file_jpg = os.path.join(DIFFUSE_MAPS_DIR, self.model_name + '.jpg')
        tex_file_png = os.path.join(DIFFUSE_MAPS_DIR, self.model_name + '.png')
        tex_file_bmp = os.path.join(DIFFUSE_MAPS_DIR, self.model_name + '.bmp')

        if os.path.exists(tex_file_jpg):
            self.tex = np.asarray(Image.open(tex_file_jpg).convert('RGB'))
        elif os.path.exists(tex_file_png):
            self.tex = np.asarray(Image.open(tex_file_png).convert('RGB'))
        elif os.path.exists(tex_file_bmp):
            self.tex = np.asarray(Image.open(tex_file_bmp).convert('RGB'))

    def get_component_distance_uv_color(self, point_samples):
        # too slow, replace it with igl
        # closest, distance_samples, face_id = trimesh.proximity.closest_point(self.mesh, point_samples)

        distance_samples, face_id, closest = igl.signed_distance(point_samples, self.vertices, self.faces)

        components = trimesh.graph.connected_component_labels(self.mesh.face_adjacency)
        component_samples = components[face_id]

        vertex_triangles = []
        for i, tri in enumerate(self.faces):
            v1, v2, v3 = self.vertices[tri]
            triangle = [v1, v2, v3]
            vertex_triangles.append(triangle)
        vertex_triangles = np.array(vertex_triangles)

        uv_triangles = []
        for i, tri in enumerate(self.faces_tex):
            v1, v2, v3 = self.vertices_tex[tri]
            triangle = [v1, v2, v3]
            uv_triangles.append(triangle)
        uv_triangles = np.array(uv_triangles)
        uv_triangles = np.dstack((uv_triangles, np.zeros((uv_triangles.shape[0], uv_triangles.shape[1]))))

        vertex_triangles = vertex_triangles[face_id]
        uv_triangles = uv_triangles[face_id]
        barycentric = trimesh.triangles.points_to_barycentric(vertex_triangles, closest)
        uv_samples = trimesh.triangles.barycentric_to_points(uv_triangles, barycentric)[:, :2]

        color_samples = uv_to_color(uv_samples, self.tex)

        return component_samples, distance_samples, uv_samples, color_samples

    def sample_on_surface(self, sample_num):
        if self.mesh is not None:
            point_samples, face_id = trimesh.sample.sample_surface_even(self.mesh, sample_num)

        return point_samples

    def sample(self, sample_mode='Importance', sample_num=64*64*64*4):
        if self.mesh is not None and self.tex is not None:
            if sample_mode == 'Uniform':
                uniform_sampler = UniformSampler(self.mesh, ratio=1.0)
                self.point_samples = uniform_sampler.sample(sample_num)
            if sample_mode == 'Importance':
                importance_sampler = ImportanceSampler(self.mesh, sample_num*10, 60)
                self.point_samples = importance_sampler.sample(sample_num)
            if sample_mode == 'Gaussian':
                self.point_samples, _ = sample_sdf_near_surface(self.mesh, number_of_points=sample_num, surface_point_method='sample')

            self.surface_points = self.sample_on_surface(sample_num=sample_num)

            self.component_samples, self.distance_samples, self.uv_samples, self.color_samples = self.get_component_distance_uv_color(self.point_samples)
