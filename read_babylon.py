import json
import os
import svgwrite
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  # required for projection='3d'
import numpy as np


def load_babylon(babylon_file):
    """Call to return the loaded babylon_file as structure of dicts, lists and values."""

    # process the path
    if not os.path.isabs(babylon_file):
        output_dir = os.path.normpath(os.path.join(os.getcwd(), babylon_file))

    with open(babylon_file) as json_file:
        return json.load(json_file)


def get_mesh_id_dict(meshes):
    """Returns a dicts of all meshes with their id as key."""

    mesh_id_dict = dict()
    for mesh in meshes:
        mesh_id_dict[mesh['id']] = mesh

    return mesh_id_dict


def get_mesh_hierarchy(meshes):
    """Returns a dict with the root meshes and their children."""

    # get the mesh id dictionary
    mesh_id_dict = get_mesh_id_dict(meshes)

    # add the children property for all meshes
    for mesh in meshes:
        if 'children' not in mesh:
            mesh['children'] = list()

    # set the children
    for mesh in meshes:
        # fill the parent field
        if mesh['parentId'] is not None:
            mesh['parent'] = mesh_id_dict[mesh['parentId']]
        else:
            mesh['parent'] = None

        # add the current mesh to its parent children
        if mesh['parentId'] is not None:
            mesh_id_dict[mesh['parentId']]['children'].append(mesh)

    # build the hierarchy
    mesh_hierarchy = list()
    for mesh in json_obj['meshes']:
        if mesh['parentId'] is None:
            mesh_hierarchy.append(mesh)

    return mesh_hierarchy


def get_mesh_positions(mesh):
    """Returns a tuple of x,y, coordinates of the mesh."""

    # define the lists
    x = list()
    y = list()
    z = list()

    # gather the data
    mesh_positions = mesh['positions']
    if mesh_positions is None:
        return np.array(x), np.array(y), np.array(z)

    vec_idx = 0
    vertex_cnt = int(len(mesh_positions) / 3)
    for i_pos in range(vertex_cnt):
        x.append(mesh_positions[vec_idx])
        y.append(mesh_positions[vec_idx + 1])
        z.append(mesh_positions[vec_idx + 2])

        vec_idx += 3

    return np.array(x), np.array(y), np.array(z)

def rotate_mesh_positions(x,y,z, rotation):
    """Returns the x,y,z coordinates which are rotated around the angles of the rotation.

    :param x:
    :param y:
    :param z:
    :param rotation:
    :return:
    """

    # fetch the rotation matrix
    R = get_rotation_matrix(rotation[2], rotation[1], rotation[0])
    p = np.zeros((3,1))
    for idx in range(len(x)):
        # store coordinates in vector
        p[0][0] = x[idx]
        p[1][0] = y[idx]
        p[2][0] = z[idx]
        # rotate vector
        p = np.dot(R, p)
        # fetch coordinate s
        x[idx] = p[0][0]
        y[idx] = p[1][0]
        z[idx] = p[2][0]

    return x,y,z

def get_transformation_matrix(translation, scaling, rotation):
    """Returns the transformation matrix based on the translation, scaling and rotation vector

    :param translation: (mandatory, 3 element numpy.ndarray) the translation vector x, y, z
    :param scaling: (mandatory, 3 element numpy.ndarray) the scaling vector x, y, z
    :param rotation: (mandatory, 3 element numpy.ndarray) the rotation vector around x, y and z axis
    :return: TransformationMatrix 3x3
    """

    # get translation matrix
    T = np.eye(4)
    T[0][3] = translation[0] # translation in x
    T[1][3] = translation[1] # translation in y
    T[2][3] = translation[2] # translation in z

    # get the scaling matrix
    S = np.eye(4)
    S[0][0] = scaling[0] # scaling along x
    S[1][1] = scaling[1] # scaling along y
    S[2][2] = scaling[2] # scaling along z

    # get the rotation matrices
    # rotation around X axis
    Rx = np.eye(4)
    Rx[1][1] = math.cos(rotation[0])
    Rx[1][2] = -math.sin(rotation[0])
    Rx[2][1] = math.sin(rotation[0])
    Rx[2][2] = math.cos(rotation[0])
    # rotation around Y axis
    Ry = np.eye(4)
    Ry[0][0] = math.cos(rotation[1])
    Ry[0][2] = math.sin(rotation[1])
    Ry[2][0] = -math.sin(rotation[1])
    Ry[2][2] = math.cos(rotation[1])
    # rotation around Z axis
    Rz = np.eye(4)
    Rz[0][0] = math.cos(rotation[2])
    Rz[0][1] = -math.sin(rotation[2])
    Rz[1][0] = math.sin(rotation[2])
    Rz[1][1] = math.cos(rotation[2])
    # build overall rotation matrix
    Ryx = np.matmul(Ry, Rx) # R = Ry * Rx
    R = np.matmul(Rz, Ryx)  # R = Rz * (Ry * Rx)

    # compute the total transformation matrix
    SR = np.matmul(S, R)   #  ST = S * T
    TSR = np.matmul(T, SR) # TSR = T * S * R = T * S * Rz * Ry * Rx

    return TSR

def get_transformation_matrix_from_mesh(mesh):
    """Returns the transformation matrix based on the mesh."""

    return get_transformation_matrix(mesh['position'], mesh['scaling'], mesh['rotation'])

def transform_coordinates(x, y, z, T):
    """Transform x, y, z coordinates with a given transformation matrix."""

    p = np.ones((4,1))
    for idx in range(len(x)):
        # build vector
        p[0][0] = x[idx]
        p[1][0] = y[idx]
        p[2][0] = z[idx]
        # transform the vector
        p = np.dot(T, p)
        # store results in coordinated
        x[idx] = p[0][0]
        y[idx] = p[1][0]
        z[idx] = p[2][0]

    return x, y, z

def get_rotation_matrix(phi = 0, theta = 0, psi = 0):
    """Returns the rotation matrix of the rotations around the angles axes.

    Rotation around Z Axes: phi
    Rotation aroung Y Axes: theta
    Rotation around X Axes: psi

    :param phi: (mandatory, float) rotation angle around the z axes in radian
    :param theta: (mandatory, float) rotation angle around the y axes in radian
    :param psi: (mandatory, float) rotation angle around the x axes in radian
    :return:
    """

    # build the rotation matrix
    R = np.zeros((3,3))
    R[0][0] = math.cos(theta) * math.cos(phi)
    R[1][0] = math.cos(theta) * math.sin(phi)
    R[2][0] = -math.sin(theta)
    R[0][1] = math.sin(psi) * math.sin(theta) * math.cos(phi) - math.cos(psi) * math.sin(phi)
    R[1][1] = math.sin(psi) * math.sin(theta) * math.sin(phi) + math.cos(psi) * math.cos(phi)
    R[2][1] = math.sin(psi) * math.cos(theta)
    R[0][2] = math.cos(psi) * math.sin(theta) * math.cos(phi) + math.sin(psi) * math.sin(phi)
    R[1][2] = math.sin(psi) * math.sin(theta) * math.sin(phi) - math.sin(psi) * math.cos(phi)
    R[2][2] = math.cos(psi) * math.cos(theta)

    return R

def print_mesh_children(mesh, level='  |- ', target=None):
    """Prints the name of the children."""

    if 'children' not in mesh:
        return

    for sub_mesh in mesh['children']:

        content_str = '{}{}'.format(level, sub_mesh['name'])
        prop_level = level.replace('|-', '|   ')
        props = ['id', 'position', 'scaling', 'rotation', 'pivotMatrix']
        for prop in props:
            if prop in sub_mesh:
                content_str += '\n{}{}: {}'.format(prop_level, prop, sub_mesh[prop])

        if target is None:
            print(content_str)
        else:
            with open(target, 'a+') as target_file:
                target_file.write('{}\n'.format(content_str))

        print_mesh_children(mesh=sub_mesh, level='  |{0}'.format(level), target=target)


def print_mesh_hierarchy(mesh_hierarchy, target=None):
    """Prints the mesh hierarchy into the console."""

    # process the path
    if not os.path.isabs(target):
        target = os.path.normpath(os.path.join(os.getcwd(), target))
    target_dir = os.path.dirname(target)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    # remove old files
    if os.path.isfile(target):
        os.remove(target)

    # print the hierarchy
    for mesh in mesh_hierarchy:

        content_str = '{}'.format(mesh['name'])
        prop_level = '  '
        props = ['id', 'position', 'scaling', 'rotation', 'pivotMatrix']
        for prop in props:
            if prop in mesh:
                content_str += '\n{}{}: {}'.format(prop_level, prop, mesh[prop])

        if target is None:
            # print the mesh to the command line
            print(content_str)
        else:
            with open(target, 'a+') as target_file:
                target_file.write('{}\n'.format(content_str))

        print_mesh_children(mesh, target=target)


def get_mesh_route(mesh, route_str=None):
    """Prints the complete route to the current mesh within the hierarchy."""

    if route_str is None:
        route_str = mesh['name']
    else:
        route_str = '{}.{}'.format(mesh['name'], route_str)

    if mesh['parent'] is not None:
        route_str = get_mesh_route(mesh['parent'], route_str=route_str)

    return route_str


def plot_meshes(mesh_list, output_dir='export', axes_to_plot=None, exclude=None, T=np.eye(4)):
    """Call to plot the meshes."""

    # process the path
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(os.getcwd(), output_dir))
    target_dir = os.path.dirname(output_dir)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    if exclude is None:
        exclude = list()

    if axes_to_plot is None:
        axes_to_plot = list()

    # plot the sub meshes
    sub_folder_cnt = dict()
    for sub_mesh in mesh_list:
        # shall the current mesh be skipped?
        if sub_mesh['name'] in exclude:
            continue

        # get full name of the mesh
        full_mesh_name = get_mesh_route(sub_mesh)

        # create new figure
        fig, ax = new_figure('{} ({})'.format(full_mesh_name, sub_mesh['id']))
        # make sure to append the axes
        cur_axes_to_plot = axes_to_plot[:]  # make sure to create a new list
        cur_axes_to_plot.append(ax)

        # calculate the current transformation matrix
        T = get_transformation_matrix_from_mesh(sub_mesh)
        # T = np.matmul(T, get_transformation_matrix_from_mesh(sub_mesh))

        # get position of mesh
        x, y, z = get_mesh_positions(sub_mesh)
        x, y, z = transform_coordinates(x, y, z, T)

        # plot data if available
        if len(x) > 0:
            print('plotting: {} ...'.format(full_mesh_name))

            # add the current plot to the mesh
            for cur_ax in cur_axes_to_plot:
                cur_scat = cur_ax.scatter(x, y, z)
                plot_wireframe(cur_ax, x, y, z, sub_mesh['indices'], color=cur_scat.get_edgecolor()[0])
        else:
            print('processing: {} ...'.format(get_mesh_route(sub_mesh)))

        # build path to output folder
        short_id = str(sub_mesh['id']).split('-')[0]
        cur_folder_name = '{}_{}'.format(sub_mesh['name'], short_id)
        if cur_folder_name not in sub_folder_cnt:
            sub_folder_cnt[cur_folder_name] = 0
        sub_folder_cnt[cur_folder_name] += 1
        if sub_folder_cnt[cur_folder_name] == 1:
            cur_sub_folder = os.path.join(output_dir, cur_folder_name)
        else:
            cur_sub_folder = os.path.join(output_dir, '{}_{}'.format(cur_folder_name, sub_folder_cnt[cur_folder_name]))

        # plot the sub meshes
        if sub_mesh['children']:
            plot_meshes(sub_mesh['children'], output_dir=cur_sub_folder, exclude=exclude, axes_to_plot=cur_axes_to_plot,
                        T=T)

        # save the figure
        file_name = os.path.join(cur_sub_folder, '{}.png'.format(sub_mesh['name']))
        save_figure(fig, file_name)

        # close the figure
        plt.close(fig)

def plot_wireframe(cur_ax, x, y, z, indices, color='k'):
    """Plots all faces as wireframe."""

    indices_idx = 0
    for idx in range(int(len(indices) / 3)):
        # get the indeces of the face
        data_idx = indices[indices_idx:indices_idx + 3]
        indices_idx += 3
        # close the loop
        data_idx = np.append(data_idx, data_idx[0])

        # build the x, y and z arrays of the current polygon
        cur_ax.plot(x[data_idx], y[data_idx], z[data_idx], color=color)


def new_figure(name=None):
    """Returns a new figure and axes."""

    my_dpi = 96
    fig = plt.figure(figsize=(2000 / my_dpi, 2000 / my_dpi), dpi=my_dpi)
    ax = fig.gca(projection='3d', adjustable='box')

    # set axes
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # set name
    if name is not None:
        fig.canvas.set_window_title(name)
        ax.set_title(name)

    return fig, ax


def save_figure(fig, file_name):
    """Saves the figure as the given filename."""

    VIEW = {'iso': (30, 30), 'x': (0, 0), 'y': (0, 90), 'z': (90, 90)}

    # process the path
    if not os.path.isabs(file_name):
        file_name = os.path.normpath(os.path.join(os.getcwd(), file_name))
    target_dir = os.path.dirname(file_name)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    # set axes
    for ax in fig.get_axes():
        set_axes_equal(ax)

    # build actual file name
    file_dir = os.path.dirname(file_name)
    file_name, file_ext = os.path.splitext(os.path.basename(file_name))

    # save the image in different perspectives
    for key, val in VIEW.items():
        # set the view for each axes
        for ax in fig.get_axes():
            ax.view_init(val[0], val[1])
        fig.savefig(os.path.join(file_dir, '{}_{}{}'.format(file_name, key, file_ext)), dpi=fig.dpi)


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


if __name__ == "__main__":
    # File I/O ######################################
    json_obj = load_babylon('Demo.babylon')

    # Gather Meshes ################################
    mesh_id_dict = get_mesh_id_dict(json_obj['meshes'])
    mesh_hierarchy = get_mesh_hierarchy(json_obj['meshes'])
    print_mesh_hierarchy(mesh_hierarchy, target='Demo_Hierarchy.txt')

    # Plot the Meshes ##############################
    plot_meshes(mesh_hierarchy, output_dir='export', exclude=['SceneSkyboxMesh', '[SceneManager]'])
    # plot_meshes([mesh_id_dict['d5c49076-6af5-48c1-913e-014a7dd8ec57']], output_dir='export', exclude=['SceneSkyboxMesh', '[SceneManager]'])
