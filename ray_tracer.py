import os
import time
import numbers
import argparse
import igl
import trimesh

import tensorflow as tf
import numpy as np

from functools import reduce
from PIL import Image
from math import sin, cos

from utils import *

tf.keras.utils.get_custom_objects().update({
    'sin': tf.math.sin
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SHAPES_DIR = os.path.join(DATA_DIR, 'shapes')
SDF_MODELS_DIR = os.path.join(DATA_DIR, 'sdf-models')
DIFFUSE_MAPS_DIR = os.path.join(DATA_DIR, 'diffuse-maps')
NORMAL_MAPS_DIR = os.path.join(DATA_DIR, 'normal-maps')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
RESULTS_COLOR_DIR = os.path.join(RESULTS_DIR, 'color-mapper')
RESULTS_UV_DIR = os.path.join(RESULTS_DIR, 'uv-mapper')
RESULTS_DECOMPOSED_UV_DIR = os.path.join(RESULTS_DIR, 'decomposed-uv-mapper')

FARAWAY = 1.0e39
EPSILON = 0.0001
MAX_MARCHING_STEPS = 255

def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)

class vec3():
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)
    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)
    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    def __abs__(self):
        return self.dot(self)
    def cross(self, b):
      return vec3(self.y * b.z - b.y * self.z,
                  self.z * b.x - b.z * self.x,
                  self.x * b.y - b.x * self.y)
    def norm(self):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))
    def components(self):
        return (self.x, self.y, self.z)
    def extract(self, cond):
        return vec3(extract(cond, self.x),
                    extract(cond, self.y),
                    extract(cond, self.z))
    def place(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r
rgb = vec3

class Sphere:
    def __init__(self, center, r, diffuse, mirror=0.):
        self.c = center
        self.r = r
        self.diffuse = diffuse
        self.mirror = mirror

    def intersect(self, O, D):
        b = 2 * D.dot(O - self.c)
        c = abs(self.c) + abs(O) - 2 * self.c.dot(O) - (self.r * self.r)
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h1 = c / h1
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h >= 0)
        return np.where(pred, h, FARAWAY)

    def diffuse_color(self, M):
        return self.diffuse

    def normal(self, M):
        return (M - self.c) * (1. / self.r)

    def light(self, O, D, d, L, E, scene, bounce):
        M = (O + D * d)
        N = self.normal(M)
        toL = (L - M).norm()
        toO = (E - M).norm()
        nudged = M + N * .0001

        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        see_light = light_distances[scene.index(self)] == light_nearest

        color = rgb(0.05, 0.05, 0.05)

        lv = np.maximum(N.dot(toL), 0)
        color += self.diffuse_color(M) * lv * see_light

        if self.mirror > 0:
            if bounce < 2:
                rayD = (D - N * 2 * D.dot(N)).norm()
                color += ray_trace(nudged, rayD, L, E, scene, bounce + 1) * self.mirror

        phong = N.dot((toL + toO).norm())
        color += rgb(.2, .2, .2) * np.power(np.clip(phong, 0, 1), 25) * see_light
        return color

class CheckeredSphere(Sphere):
    def diffuse_color(self, M):
        checker = ((M.x * 2).astype(int) % 2) == ((M.z * 2).astype(int) % 2)
        return self.diffuse * checker

class Mesh:
    def __init__(self, model_name, center, diffuse, use_texture=True, mirror=0.):
        self.model_name = model_name

        self.c = center
        self.diffuse = diffuse
        self.mirror = mirror

        self.use_texture = use_texture

        self.mesh = self.get_mesh(self.model_name)

    def get_mesh(self, model_name):
        model = trimesh.load(os.path.join(SHAPES_DIR, model_name + '.obj'))

        f = open(os.path.join(SHAPES_DIR, model_name + '.obj'), 'r')
        text = f.read()
        text = trimesh.util.decode_text(text)
        text = '\n{}\n'.format(text.strip().replace('\r\n', '\n'))
        text = text.replace('\\\n', '')

        vertices, vertices_norm, vertices_tex, vc = trimesh.exchange.obj._parse_vertices(text=text)

        face_tuples = trimesh.exchange.obj._preprocess_faces(text=text)
        material, current_object, chunk = face_tuples.pop()
        face_lines = [i.split('\n', 1)[0] for i in chunk.split('\nf ')[1:]]
        joined = ' '.join(face_lines).replace('/', ' ')
        array = np.fromstring(joined, sep=' ', dtype=np.int64) - 1
        columns = len(face_lines[0].strip().replace('/', ' ').split())
        faces, faces_tex, faces_norm = trimesh.exchange.obj._parse_faces_vectorized(
            array=array, columns=columns, sample_line=face_lines[0])

        return dict(model=model, vertices_tex=vertices_tex, faces_tex=faces_tex)

    def intersect_unit_sphere(self, O, D):
        b = 2 * D.dot(O - self.c)
        c = abs(self.c) + abs(O) - 2 * self.c.dot(O) - (1. * 1.)
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h1 = c / h1
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h >= 0)
        return np.where(pred, h, FARAWAY)

    def intersect(self, O, D):
        depth = self.intersect_unit_sphere(O, D)
        dist = np.full(depth.shape, 0.)
        mask = ~(depth == FARAWAY)
        if depth[mask].size > 0:
            for i in range(MAX_MARCHING_STEPS):
                march = np.dstack(((O - self.c) + D * depth).components())[0][mask]
                dist[mask], _, _ = igl.signed_distance(march, self.mesh['model'].vertices, self.mesh['model'].faces)
                out_of_range = dist > 1. - EPSILON
                depth[out_of_range] = FARAWAY
                intersected = dist < EPSILON
                mask = ~(out_of_range | intersected)
                if depth[mask].size > 0:
                    depth[mask] += dist[mask]
                else:
                    return depth
        return depth

    def diffuse_color(self, M):
        return self.diffuse

    def get_coler(self, M):
        M = np.dstack((M - self.c).components())[0]

        _, face_id, closest = igl.signed_distance(M, self.mesh['model'].vertices, self.mesh['model'].faces)

        vertex_triangles = []
        for i, tri in enumerate(self.mesh['model'].faces):
            v1, v2, v3 = self.mesh['model'].vertices[tri]
            triangle = [v1, v2, v3]
            vertex_triangles.append(triangle)
        vertex_triangles = np.array(vertex_triangles)

        uv_triangles = []
        for i, tri in enumerate(self.mesh['faces_tex']):
            v1, v2, v3 = self.mesh['vertices_tex'][tri]
            triangle = [v1, v2, v3]
            uv_triangles.append(triangle)
        uv_triangles = np.array(uv_triangles)
        uv_triangles = np.dstack((uv_triangles, np.zeros((uv_triangles.shape[0], uv_triangles.shape[1]))))

        vertex_triangles = vertex_triangles[face_id]
        uv_triangles = uv_triangles[face_id]
        barycentric = trimesh.triangles.points_to_barycentric(vertex_triangles, closest)
        uv = trimesh.triangles.barycentric_to_points(uv_triangles, barycentric)[:, :2]

        tex = np.asarray(Image.open(os.path.join(DIFFUSE_MAPS_DIR, self.model_name + '.jpg')))

        color = uv_to_color(uv, tex)
        return vec3(color[:,0], color[:,1], color[:,2])

    def normal(self, M):
        M = np.dstack((M - self.c).components())[0]
        _, face_id, _ = igl.signed_distance(M, self.mesh['model'].vertices, self.mesh['model'].faces)
        N = self.mesh['model'].face_normals[face_id]
        return vec3(N[:,0], N[:,1], N[:,2]).norm()

    def light(self, O, D, d, L, E, scene, bounce):
        M = (O + D * d)
        N = self.normal(M)
        toL = (L - M).norm()
        toO = (E - M).norm()
        nudged = M + N * .0001

        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        see_light = light_distances[scene.index(self)] == light_nearest

        color = rgb(0.05, 0.05, 0.05)

        lv = np.maximum(N.dot(toL), 0)
        if self.use_texture:
            color += self.get_coler(M) * lv * see_light
        else:
            color += self.diffuse_color(M) * lv * see_light

        if self.mirror > 0:
            if bounce < 2:
                rayD = (D - N * 2 * D.dot(N)).norm()
                color += ray_trace(nudged, rayD, L, E, scene, bounce + 1) * self.mirror

        phong = N.dot((toL + toO).norm())
        color += rgb(.2, .2, .2) * np.power(np.clip(phong, 0, 1), 25) * see_light
        return color

class SDF:
    def __init__(self, model_name, center, scale, theta, diffuse, texture_model_type, use_normal_map=False, mirror=0.):
        self.model_name = model_name

        self.c = center
        self.s = scale
        self.diffuse = diffuse
        self.mirror = mirror

        t = np.radians(theta)
        c, s = np.cos(t), np.sin(t)
        self.r = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))

        self.sdf_model = self.get_sdf_model(self.model_name)

        self.texture_model_type = texture_model_type

        self.use_normal_map = use_normal_map

        if self.texture_model_type == 'color':
            self.texture_model = self.get_color_model(self.model_name)
        elif self.texture_model_type == 'uv':
            self.texture_model = self.get_uv_model(self.model_name)
        elif self.texture_model_type == 'uv_decomposed':
            self.texture_model = self.get_decomposed_uv_model(self.model_name)

    def get_sdf_model(self, model_name):
        model = tf.keras.models.model_from_json(open(os.path.join(SDF_MODELS_DIR, model_name + '.json'), 'r').read())
        model.load_weights(os.path.join(SDF_MODELS_DIR, model_name + '.h5'))
        return model

    def get_color_model(self, model_name):
        model = tf.keras.models.model_from_json(open(os.path.join(RESULTS_COLOR_DIR, model_name + '.json'), 'r').read())
        model.load_weights(os.path.join(RESULTS_COLOR_DIR, model_name + '.h5'))
        return model

    def get_uv_model(self, model_name):
        model = tf.keras.models.model_from_json(open(os.path.join(RESULTS_UV_DIR, model_name + '.json'), 'r').read())
        model.load_weights(os.path.join(RESULTS_UV_DIR, model_name + '.h5'))
        return model

    def get_decomposed_uv_model(self, model_name):
        model = tf.keras.models.model_from_json(open(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '.json'), 'r').read())
        model.load_weights(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '.h5'))
        return model

    def intersect_unit_sphere(self, O, D):
        b = 2 * D.dot(O - self.c)
        c = abs(self.c) + abs(O) - 2 * self.c.dot(O) - (self.s * self.s)
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h1 = c / h1
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h >= 0)
        return np.where(pred, h, FARAWAY)

    def signed_distance(self, p):
        p = (p - np.array(self.c.components())) @ self.r.T
        return self.sdf_model.predict(p/self.s).squeeze() * self.s

    def intersect(self, O, D):
        if np.isscalar(O.x):
            O_components = np.tile(np.array(O.components()), (D.x.shape[0],1))
            O = vec3(O_components[:,0], O_components[:,1], O_components[:,2])
        d_vec = O - self.c
        dist_to_center = np.sqrt(d_vec.dot(d_vec))
        out_of_sphere = dist_to_center > self.s + EPSILON
        depth = self.intersect_unit_sphere(O, D)
        depth[~out_of_sphere] = 0
        dist = np.full(depth.shape, 0.)
        mask = ~(depth == FARAWAY)
        if depth[mask].size > 0:
            dist_to_center = np.full(depth.shape, FARAWAY)
            for i in range(MAX_MARCHING_STEPS):
                d_vec = (O + D * depth) - self.c
                dist_to_center[mask] = np.sqrt(d_vec.dot(d_vec)[mask])
                out_of_sphere = dist_to_center > self.s + EPSILON
                depth[out_of_sphere] = FARAWAY
                mask = ~out_of_sphere
                if np.count_nonzero(mask) > 0:
                    march = np.dstack((O + D * depth).components())[0][mask]
                    dist[mask] = self.signed_distance(march)
                    out_of_range = dist > self.s + EPSILON
                    depth[out_of_range] = FARAWAY
                    intersected = dist < EPSILON
                    mask = ~(out_of_range | intersected)
                    if depth[mask].size > 0:
                        depth[mask] += dist[mask]
                    else:
                        return depth
        return depth

    def diffuse_color(self, M):
        return self.diffuse

    def infer_coler(self, M):
        M = np.dstack(M.components())[0]
        M = (M - np.array(self.c.components())) @ self.r.T
        color = self.texture_model.predict(M/self.s)
        return vec3(color[:,0], color[:,1], color[:,2])

    def infer_uv(self, M):
        M = np.dstack(M.components())[0]
        M = (M - np.array(self.c.components())) @ self.r.T
        uv = self.texture_model.predict(M/self.s)
        return uv

    def lookup_color(self, uv):
        tex_file_jpg = os.path.join(DIFFUSE_MAPS_DIR, self.model_name + '.jpg')
        tex_file_png = os.path.join(DIFFUSE_MAPS_DIR, self.model_name + '.png')
        tex_file_bmp = os.path.join(DIFFUSE_MAPS_DIR, self.model_name + '.bmp')

        if os.path.exists(tex_file_jpg):
            tex = np.asarray(Image.open(tex_file_jpg).convert('RGB'))
        elif os.path.exists(tex_file_png):
            tex = np.asarray(Image.open(tex_file_png).convert('RGB'))
        elif os.path.exists(tex_file_bmp):
            tex = np.asarray(Image.open(tex_file_bmp).convert('RGB'))

        color = uv_to_color(uv, tex)

        return vec3(color[:,0], color[:,1], color[:,2])

    def normal_map(self, uv):
        tex_file_jpg = os.path.join(NORMAL_MAPS_DIR, self.model_name + '.jpg')
        tex_file_png = os.path.join(NORMAL_MAPS_DIR, self.model_name + '.png')
        tex_file_bmp = os.path.join(NORMAL_MAPS_DIR, self.model_name + '.bmp')

        if os.path.exists(tex_file_jpg):
            tex = np.asarray(Image.open(tex_file_jpg).convert('RGB'))
        elif os.path.exists(tex_file_png):
            tex = np.asarray(Image.open(tex_file_png).convert('RGB'))
        elif os.path.exists(tex_file_bmp):
            tex = np.asarray(Image.open(tex_file_bmp).convert('RGB'))

        normal = uv_to_normal(uv, tex)

        return vec3(normal[:,0], normal[:,1], normal[:,2])

    def normal(self, M):
        M = np.dstack(M.components())[0]

        a = np.column_stack((M[:,0] + EPSILON, M[:,1], M[:,2]))
        b = np.column_stack((M[:,0] - EPSILON, M[:,1], M[:,2]))
        c = np.column_stack((M[:,0], M[:,1] + EPSILON, M[:,2]))
        d = np.column_stack((M[:,0], M[:,1] - EPSILON, M[:,2]))
        e = np.column_stack((M[:,0], M[:,1], M[:,2] + EPSILON))
        f = np.column_stack((M[:,0], M[:,1], M[:,2] - EPSILON))

        x = self.signed_distance(a) - self.signed_distance(b)
        y = self.signed_distance(c) - self.signed_distance(d)
        z = self.signed_distance(e) - self.signed_distance(f)

        N = np.column_stack((x, y, z))

        return vec3(N[:,0], N[:,1], N[:,2]).norm()

    def light(self, O, D, d, L, E, scene, bounce):
        M = (O + D * d)

        if self.texture_model_type == 'color':
            surface_color = self.infer_coler(M)
        elif self.texture_model_type == 'uv' or self.texture_model_type == 'uv_decomposed':
            surface_uv = self.infer_uv(M)
            surface_color = self.lookup_color(surface_uv)
        else:
            surface_color = self.diffuse_color(M)

        N = self.normal(M)

        if self.use_normal_map:
            T = N.cross(vec3(0.0, 0.0, 1.0)).norm()
            B = N.cross(T).norm()
            N_map = self.normal_map(surface_uv)
            Nx = T.x*N_map.x + B.x*N_map.y + N.x*N_map.z
            Ny = T.y*N_map.x + B.y*N_map.y + N.y*N_map.z
            Nz = T.z*N_map.x + B.z*N_map.y + N.z*N_map.z
            N = vec3(Nx, Ny, Nz)

        toL = (L - M).norm()
        toO = (E - M).norm()
        nudged = M + N * .0001

        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        see_light = light_distances[scene.index(self)] == light_nearest

        color = rgb(0.05, 0.05, 0.05)

        lv = np.maximum(N.dot(toL), 0)
        color += surface_color * lv * see_light

        if self.mirror > 0:
            if bounce < 2:
                rayD = (D - N * 2 * D.dot(N)).norm()
                color += ray_trace(nudged, rayD, L, E, scene, bounce + 1) * self.mirror

        phong = N.dot((toL + toO).norm())
        color += rgb(.2, .2, .2) * np.power(np.clip(phong, 0, 1), 25) * see_light
        return color

def ray_trace(O, D, L, E, scene, bounce=0):
    distances = [s.intersect(O, D) for s in scene]
    nearest = reduce(np.minimum, distances)
    color = rgb(0, 0, 0)
    for (s, d) in zip(scene, distances):
        hit = (nearest != FARAWAY) & (d == nearest)
        if np.any(hit):
            dc = extract(hit, d)
            Oc = O.extract(hit)
            Dc = D.extract(hit)
            cc = s.light(Oc, Dc, dc, L, E, scene, bounce)
            color += cc.place(hit)
    color.x[color.x==0] = 1.
    color.y[color.y==0] = 1.
    color.z[color.z==0] = 1.
    return color

def look_at(E):
    camera_position = np.array([E.x, E.y, E.z])
    camera_target = np.array([0, 0, 0])
    up_vector = np.array([0, 1, 0])

    vector = camera_position - camera_target
    vector = vector / np.linalg.norm(vector)

    vector2 = np.cross(up_vector, vector)
    vector2 = vector2 / np.linalg.norm(vector2)

    vector3 = np.cross(vector, vector2)
    return np.array([
        [vector2[0], vector3[0], vector[0], 0.0],
        [vector2[1], vector3[1], vector[1], 0.0],
        [vector2[2], vector3[2], vector[2], 0.0],
        [-np.dot(vector2, camera_position), -np.dot(vector3, camera_position), np.dot(vector, camera_position), 1.0]
    ])

def render(w, h, x0, y0, x1, y1, L, E, scene, image_file):
    x = np.tile(np.linspace(x0, x1, w), h)
    y = np.repeat(np.linspace(y0, y1, h), w)

    Q = np.column_stack((x, y, np.full(x.shape, 0.)))
    lookat_matrix = look_at(E).T
    Q = Q @ lookat_matrix[:3, :3]

    t0 = time.time()
    Q = vec3(Q[:,0], Q[:,1], Q[:,2])
    color = rgb(0, 0, 0)
    color += ray_trace(E, (Q - E).norm(), L, E, scene)
    print('Took', time.time() - t0)

    crgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), 'L') for c in color.components()]
    Image.merge('RGB', crgb).save(image_file)
