import os
import numpy as np
import tensorflow as tf

def uv_to_color(uv, tex, normalize=True):
    uv_pix = uv.copy()
    uv_pix[:,1] *= (tex.shape[0]-1)
    uv_pix[:,0] *= (tex.shape[1]-1)
    uv_pix = np.floor(uv_pix).astype(np.int16)
    color = tex[tex.shape[0] - (uv_pix[:,1]+1), uv_pix[:,0]]
    if normalize:
        return color / 255
    else:
        return color

def uv_to_normal(uv, tex):
    uv_pix = uv.copy()
    uv_pix[:,1] *= (tex.shape[0]-1)
    uv_pix[:,0] *= (tex.shape[1]-1)
    uv_pix = np.floor(uv_pix).astype(np.int16)
    normal = tex[tex.shape[0] - (uv_pix[:,1]+1), uv_pix[:,0]] / 255
    normal = (2.0*normal) - 1.0
    return normal

def compute_unit_sphere_transform(mesh):
    translation = -mesh.bounding_box.centroid
    scale = 1 / np.max(np.linalg.norm(mesh.vertices + translation, axis=1))
    return translation, scale

# re-implemented with tensorflow
# def get_fourier_features(x, fourier_max_freq):
#     bvals = 2.**np.arange(fourier_max_freq/2)
#     bvals = np.reshape(np.eye(3)*bvals[:,None,None], [len(bvals)*3, 3])
#     avals = np.ones((bvals.shape[0]))

#     x = np.reshape(x, [-1,3])

#     x_fourier = np.concatenate([avals * np.sin(x @ bvals.T), 
#         avals * np.cos(x @ bvals.T)], axis=-1)

#     return x_fourier

def get_fourier_features(x, fourier_max_freq, batch_size=0, dim=3):
    import tensorflow as tf

    bvals = 2.**np.arange(fourier_max_freq/2)
    bvals = np.reshape(np.eye(dim)*bvals[:,None,None], [len(bvals)*dim, dim])
    avals = np.ones((bvals.shape[0]))

    if batch_size <= 0:
        x = tf.reshape(x, [-1,dim])
    else:
        x = tf.reshape(x, [-1,batch_size,dim])

    x_fourier = tf.concat([avals * tf.math.sin(tf.linalg.matmul(x, bvals.T)), 
        avals * tf.math.cos(tf.linalg.matmul(x, bvals.T))], axis=-1)

    return x_fourier

def write_ply(file_name, points, color):
    with open(file_name, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % points.shape[0])
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for i in range(points.shape[0]):
            x = points[i][0]
            y = points[i][1]
            z = points[i][2]
            r = color[i][0]
            g = color[i][2]
            b = color[i][1]
            f.write('%g %g %g %d %d %d\n'% (x, y, z, r, b, g))

def compute_accuracy_precision(predictions, true_values):
    N = true_values.shape[0]
    accuracy = (true_values == predictions).sum() / N
    TP = ((predictions == 1) & (true_values == 1)).sum()
    FP = ((predictions == 1) & (true_values == 0)).sum()
    precision = TP / (TP+FP)

    return accuracy, precision
