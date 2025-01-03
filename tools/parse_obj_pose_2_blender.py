"""
this script provides a simple demo on how to interpret the object pose parameters saved in obj_fit.pkl file

Usage: python tools/parse_obj_pose.py -s [path to a sequence]

Author: Xianghui Xie
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import sys, os

import numpy as np

sys.path.append(os.getcwd())
import os.path as osp
import pickle as pkl
from scipy.spatial.transform import Rotation
# from psbody.mesh import Mesh
import trimesh
from trimesh import Trimesh
from data.frame_data import FrameDataReader
from data.const import USE_PSBODY

import json
from ipdb import set_trace as st
from os.path import join, basename, dirname, isfile

# path to the simplified mesh used for registration
simplified_mesh = {
    "backpack":"backpack/backpack_f1000.ply",
    'basketball':"basketball/basketball_f1000.ply",
    'boxlarge':"boxlarge/boxlarge_f1000.ply",
    'boxtiny':"boxtiny/boxtiny_f1000.ply",
    'boxlong':"boxlong/boxlong_f1000.ply",
    'boxsmall':"boxsmall/boxsmall_f1000.ply",
    'boxmedium':"boxmedium/boxmedium_f1000.ply",
    'chairblack': "chairblack/chairblack_f2500.ply",
    'chairwood': "chairwood/chairwood_f2500.ply",
    'monitor': "monitor/monitor_closed_f1000.ply",
    'keyboard':"keyboard/keyboard_f1000.ply",
    'plasticcontainer':"plasticcontainer/plasticcontainer_f1000.ply",
    'stool':"stool/stool_f1000.ply",
    'tablesquare':"tablesquare/tablesquare_f2000.ply",
    'toolbox':"toolbox/toolbox_f1000.ply",
    "suitcase":"suitcase/suitcase_f1000.ply",
    'tablesmall':"tablesmall/tablesmall_f1000.ply",
    'yogamat': "yogamat/yogamat_f1000.ply",
    'yogaball':"yogaball/yogaball_f1000.ply",
    'trashbin':"trashbin/trashbin_f1000.ply",
}




# def transform_camera_poses(relative_camera_poses, scale_factor, translation_vector):
#     """
#     Transform the relative camera poses to maintain consistency after bbox normalization.
#     :param relative_camera_poses: List of dictionaries with 'cameraRot' and 'cameraTrans'
#     :param scale_factor: Scaling factor applied to the bounding box
#     :param translation_vector: Translation vector applied to the bounding box
#     :return: Transformed relative_camera_poses
#     """
    
#     # T = np.array([
#     #     [1, 0, 0], 
#     #     [0, -1, 0], 
#     #     [0, 0, -1]
#     #    ]) # opencv to blender
#     T = np.array([
#         [0, 0, 1], 
#         [0, 1, 0], 
#         [-1, 0, 0]
#         ]) # opencv to blender

#     transformed_poses = []
    
    
#     config_folder = "/home/xuyimeng/Data/behave-dataset/dataset/behave/sequences/Date01_Sub01_backpack_hand/../../calibs/Date01/config"
#     kids = [0,1,2,3]

#     def load_kinect_poses(config_folder, kids):
#         pose_calibs = [json.load(open(join(config_folder, f"{x}/config.json"))) for x in kids]
#         rotations = [np.array(pose_calibs[x]['rotation']).reshape((3, 3)) for x in kids]
#         translations = [np.array(pose_calibs[x]['translation']) for x in kids]
#         return rotations, translations
    
#     all_rotations, all_translations = load_kinect_poses(config_folder, kids) # c2w
#     # rotations, translations = all_rotations[0], all_translations[0]
    

#     # for pose in relative_camera_poses:
#     #     cameraRot = np.array(pose['cameraRot'])
#     #     cameraTrans = np.array(pose['cameraTrans'])
    
#     for cameraRot, cameraTrans in zip(all_rotations, all_translations):
        
#         # Apply translation and scaling to camera translation
#         cameraTrans = (cameraTrans + translation_vector) * scale_factor
        

#         cameraRot = T @ cameraRot
#         cameraTrans = T @ cameraTrans
#         # print("transform_camera_poses", cameraRot.shape, cameraTrans.shape)
        

#         transformed_poses.append({
#             'cameraRot': cameraRot.tolist(),
#             'cameraTrans': cameraTrans.tolist()
#         })

#     return transformed_poses



def transform_camera_poses(relative_camera_poses, scale_factor, translation_vector):
    """
    Transform the relative camera poses to maintain consistency after bbox normalization.
    :param relative_camera_poses: List of dictionaries with 'cameraRot' and 'cameraTrans'
    :param scale_factor: Scaling factor applied to the bounding box
    :param translation_vector: Translation vector applied to the bounding box
    :return: Transformed relative_camera_poses
    """
    
    T = np.array([
        [1, 0, 0], 
        [0, 0, 1], 
        [0, -1, 0]
       ]) # opencv to blender
    # T = np.array([
    #     [-1, 0, 0], 
    #     [0, -1, 0], 
    #     [0, 0, 1]
    #    ]) # opencv to blender
    # T = np.array([
    #     [0, 0, 1], 
    #     [0, 1, 0], 
    #     [-1, 0, 0]
    #     ]) # opencv to blender


    # T = np.array([
    #     [-1, 0, 0], 
    #     [0, 1, 0], 
    #     [0, 0, -1]
    #    ]) # pytorch3d to blender
    print("# pytorch3d to blender", T)


    transformed_poses = []

    for pose in relative_camera_poses:
        cameraRot = np.array(pose['cameraRot'])
        cameraTrans = np.array(pose['cameraTrans'])
        
        
        # Apply translation and scaling to camera translation
        cameraTrans = (cameraTrans + translation_vector) * scale_factor
        
        # Transform rotation and translation
        # opengl_to_blender
        # print(cameraRot.shape)
        cameraRot = T @ cameraRot
        
        cameraTrans = T @ cameraTrans
        # print("transform_camera_poses", cameraRot.shape, cameraTrans.shape)
        

        transformed_poses.append({
            'cameraRot': cameraRot.tolist(),
            'cameraTrans': cameraTrans.tolist()
        })

    return transformed_poses



def normalize_object_bbox(objCorners3DRest):
    """
    Normalize the bounding box to [-1, 1] along its longest side.
    :param objCorners3DRest: Array of object bounding box corners (8, 3)
    :return: normalized_corners, scale_factor, translation_vector
    """
    # Compute the bounding box dimensions
    min_corner = np.min(objCorners3DRest, axis=0)
    max_corner = np.max(objCorners3DRest, axis=0)
    bbox_dims = max_corner - min_corner
    
    print("min_corner, max_corner:", min_corner, max_corner)

    # # Find the longest side and compute the scaling factor
    # longest_side = np.max(bbox_dims)
    # scale_factor = 1 / longest_side  # Scale longest side to [-0.5,0.5]
  
    # Compute the bounding box diagonal length
    bbox_diagonal = np.linalg.norm(bbox_dims)
    scale_factor = 1 / bbox_diagonal

    # Compute the translation vector to center the bbox at origin
    bbox_center = (min_corner + max_corner) / 2
    translation_vector = -bbox_center

    # Normalize the corners
    normalized_corners = (objCorners3DRest + translation_vector) * scale_factor

    return normalized_corners, scale_factor, translation_vector


def main(args):
    reader = FrameDataReader(args.seq_folder, check_image=False)
    category = reader.seq_info.get_obj_name(True)

    # temp_simp, temp_full = Mesh(), Mesh()
    name = reader.seq_info.get_obj_name()
    # load simplified mesh template (the mesh used for registration), make sure to set process=False to avoid reordering vertices
    temp_simp: Trimesh = trimesh.load_mesh(osp.join(args.seq_folder, f"../../objects/{simplified_mesh[name]}"), process=False)
    # load full template mesh
    temp_full = trimesh.load_mesh(osp.join(args.seq_folder, f"../../objects/{name}/{name}.obj"), process=False)
    print("full obj model:", osp.join(args.seq_folder, f"../../objects/{name}/{name}.obj"))
    # st()
    # center the meshes
    center = np.mean(temp_simp.vertices, 0)
    temp_simp.vertices -= center
    temp_full.vertices -= center

    # frames = np.random.choice(range(0, len(reader)), 5, replace=False)
    # frames = [0, 1, 2, 3]
    frames = np.arange(30)
    outfolder = osp.join(f'tmp/{reader.seq_name}')
    os.makedirs(outfolder, exist_ok=True)
    
    relative_camera_poses = []
    
    normalized_corners, scale_factor, translation_vector = normalize_object_bbox(temp_full.vertices)
    
    for idx in frames:
        idx = int(idx)
        pose_file = osp.join(reader.get_frame_folder(idx), category, f'{args.obj_name}/{category}_fit.pkl')
        data = pkl.load(open(pose_file, 'rb'))
        angle, trans = data['angle'], data['trans']
        rot = Rotation.from_rotvec(angle).as_matrix()

        # transform canonical mesh to fitting
        temp_simp_transformed = Trimesh(np.array(temp_simp.vertices), np.array(temp_simp.faces), process=False)
        temp_simp_transformed.vertices = np.matmul(temp_simp_transformed.vertices, rot.T) + trans
        temp_full_transformed = Trimesh(np.array(temp_full.vertices), np.array(temp_full.faces), process=False)
        # temp_full_transformed.v = np.matmul(temp_full_transformed.vertices, rot.T) + trans
        temp_full_transformed.vertices = np.matmul(temp_full_transformed.vertices, rot.T) + trans

        obj_fit = reader.get_objfit(idx, args.obj_name)

        if USE_PSBODY:
            obj_fit.write_ply(osp.join(outfolder, f'{reader.frame_time(idx)}_fit.ply'))
            ov_gt = obj_fit.v
        else:
            # use trimesh
            obj_fit.export(osp.join(outfolder, f'{reader.frame_time(idx)}_fit.ply'))
            ov_gt = obj_fit.vertices
        temp_full_transformed.export(osp.join(outfolder, f'{reader.frame_time(idx)}_full_transformed.ply'))
        temp_simp_transformed.export(osp.join(outfolder, f'{reader.frame_time(idx)}_simp_transformed.ply'))
        assert np.sum((ov_gt-temp_simp_transformed.vertices)**2) < 1e-8
        
        # add caemra traj
        cameraRot = rot.T
        cameraTrans = - np.matmul(rot.T, trans) + center # - rot.T @ trans
        # print(cameraTrans.shape)
        # st()
        
        relative_camera_poses.append({
            'cameraRot': cameraRot.tolist(),
            'cameraTrans': cameraTrans.tolist()
        })
    
    print("Total camera poses:", len(relative_camera_poses))
    
     # Transform the relative camera poses
    relative_camera_poses = transform_camera_poses(relative_camera_poses, scale_factor, translation_vector)
    print("Total camera poses after transform:", len(relative_camera_poses))
    # # Save the relative camera poses to a file
    output_file = os.path.join(f"tmp/{reader.seq_name}", 'relative_camera_poses_v3_blender_scaled.json')
    with open(output_file, 'w') as f:
        json.dump(relative_camera_poses, f)
    print(output_file)
          
    print(f'files saved to tmp/{reader.seq_name}')
    print('all done')
    

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-s', '--seq_folder')
    parser.add_argument('-on', '--obj_name', help='object fitting save name, for final dataset, use fit01',
                        default='fit01')

    args = parser.parse_args()

    main(args)