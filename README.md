# BEHAVE dataset (CVPR'22)
[[ArXiv]](https://arxiv.org/abs/2204.06950) [[Project Page]](http://virtualhumans.mpi-inf.mpg.de/behave)
<p align="center">
<img src="images/dataset.png" alt="teaser" width="1920"/>
</p>
BEHAVE is a dataset for full-body human-object interactions captured in natural environments. We provide multi-view RGBD frames and corresponding 3D SMPL and object fits along with the annotated contacts between them.  

## Contents
1. [Dependencies](#dependencies)
2. [Dataset Structure](#dataset-structure)
3. [Example usage](#example-usage)
   - [Generate contact labels](#generate-contact-labels)
   - [Visualize GT data](#Visualize-GT-data)
   - [Parse object pose parameters](#parse-object-pose-parameters)
   - [Parse SMPL pose parameters](#parse-smpl-pose-parameters)
   - [Generate images from raw videos](#generate-images-from-raw-videos) 
   - [Generate point clouds from RGBD images](#generate-point-clouds-from-RGBD-images)
4. [License](#license)
5. [Citation](#citation)

Code for BEHAVE model can be accesed here: https://github.com/bharat-b7/BEHAVE


## Dependencies
The code is tested on `python 3.7, Debian 10`.

To start with, create a conda environment: `conda create -n behave python=3.7`, and then `conda activate behave`

Most dependencies can be installed with: `pip install -r requirements.txt`

Some external libraries need to be installed manually:
1. psbody mesh library. See [installation](https://github.com/MPI-IS/mesh#installation). Alternatively, one can use `trimesh` library by set `USE_PSBODY=False` in `data/const.py`. 
2. igl. `conda install -c conda-forge igl` This is used to compute contacts. 
3. pytorch3d: `conda install -c fvcore -c iopath -c conda-forge fvcore iopath`, and then `conda install -c pytorch3d pytorch3d`. 


## Dataset Structure
After unzip the dataset, you can find three subfolders: `calibs`, `objects`, `sequences`. The summary of each folder is described below:
```
calibs: Kinect camera intrinsics and extrinsics for different locations
objects: 3D scans of the 20 objects
sequences: color, depth paired with SMPL and object fits of human-object interaction sequences
split.json: train and test split
```
We discuss details of each folder next:

**calibs**: This folder stores the calibrations of Kinects.

```
DATASET_PATH
|--calibs           # Kinect camera intrinsics and extrinsics for different locations
|----Date[xx]       # background and camera poses for the scene on this date
|------background   # background image and point cloud 
|------config       # camera poses
|---intrinsics      # intrinsics of 4 kinect camera
```

**objects**: This folder provides the scans of our template objects. 
```
DATASET_PATH
|--objects
|----object_name
|------object_name.jpg  # one photo of the object
|------object_name.obj  # reconstructed 3D scan of the object
|------object_name.obj.mtl  # mesh material property
|------object_name_tex.jpg  # mesh texture
|------object_name_fxxx.ply  # simplified object mesh 
```

**sequences**: This folder provides multi-view RGB-D images and SMPL, object registrations.
```
DATASET_PATH
|--sequences
|----sequence_name
|------info.json  # a file storing the calibration information for the sequence
|------t*.000     # one frame folder
|--------k[0-3].color.jpg           # color images of the frame
|--------k[0-3].depth.png           # depth images 
|--------k[0-3].person_mask.jpg     # human masks
|--------k[0-3].obj_rend_mask.jpg   # object masks
|--------k[0-3].color.json          # openpose detections
|--------k[0-3].mocap.[json|ply]    # FrankMocap estimated pose and mesh
|--------person
|----------person.ply               # segmented person point cloud
|----------fit02                    # registered SMPL mesh and parameters
|--------object_name
|----------object_name.ply          # segmented object point cloud
|----------fit01                    # object registrations
```
Note: we store the SMPL-H parameters and corresponding mesh inside each `fit02` folder. If you would like to use other body models e.g. SMPL or SMPL-X, please refer to [this repo](https://github.com/vchoutas/smplx/tree/master/transfer_model) for conversions between different body models. 

**split.json**: this file provides the official train and test split for the dataset. The split is based on sequence name. In total there are 231 sequences for training and 90 sequences for testing. 


## Example usage
Here we describe some example usages of our dataset: 

### Generate contact labels

We provide sample code in `compute_contacts.py` to generate contact labels from SMPL and object registrations. Run with:
```
python compute_contacts.py -s BEHAVE_PATH/sequences/TARGET_SEQ 
```
It samples 10k points on the object surface and compute binary contact label, and the correspondence SMPL vertices for each point. The result is saved as an `npz` file in the same folder of object registration results. 

### Visualize GT data

We provide example code in `behave_demo.py` that shows how to access different annotations provided in our dataset. It also renders the SMPL and object registration of a given sequence. Once you have the dataset and dependencies ready, run:
```
python behave_demo.py -s BEHAVE_PATH/sequences/Date04_Sub05_boxlong -v YOUR_VISUALIZE_PATH -vc 
```
you should be able to see this video inside `YOUR_VISUALIZE_PATH`:
<p align="center">
<img src="images/demo_out.gif" alt="sample" width="100%"/>
</p>


### Parse object pose parameters

The object registration parameters are saved as axis angle and translation in file `[obj_name]_fit.pkl`. These parameters transform the *centered* canonical templates to the Kinect camera coordinate. We provide a simple script in `tools/parse_obj_pose.py` to show how to use these parameters:

```
python tools/parse_obj_pose.py -s [the path to a BEHAVE sequence]
```
after runing this, you can see the transformed meshes stored under the folder `tmp/[sequence name]`. 

### Parse SMPL pose parameters

We use the [SMPL-H](https://mano.is.tue.mpg.de/) body model, please download the latest model(v1.2) from the [website](https://mano.is.tue.mpg.de/). To convert saved SMPL parameters to mesh, check the [example script](tools/smpl_params2mesh.py).


### Generate images from raw videos
```shell
python tools/video2images.py [path to one video] [output path] 
```
The provided depth images are already pixel aligned with color images, i.e. depth images have the same resolution as color images. 

We also release the registrations for all the frames, which you can find download links in [our website](https://virtualhumans.mpi-inf.mpg.de/behave/). Note that NOT all frames have registrations aligned with RGBD images. Some frames are too difficult to obtain accurate registrations hence we discard them or split one sequence into two. You can find the correspondence from each parameter to RGBD images by the `frame_times` information stored in each `npz` file (30fps annotations files). 

### Generate point clouds from RGBD images
```shell
python tools/rgbd2pclouds.py BEHAVE_SEQ_ROOT/Date05_Sub05_chairwood -t obj
```
By default, the generated point clouds will be saved to the same directory of the provided sequence path. 

## License
Please read the LICENSE file carefully. 


## Citation
If you use our code or data, please cite:
```bibtex
@inproceedings{bhatnagar22behave,
  title={Behave: Dataset and method for tracking human object interactions},
  author={Bhatnagar, Bharat Lal and Xie, Xianghui and Petrov, Ilya A and Sminchisescu, Cristian and Theobalt, Christian and Pons-Moll, Gerard},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15935--15946},
  year={2022}
}
```
