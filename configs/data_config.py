
"""_________________________________data_loader.py_________________________________________"""
data_root = 'datasets/'

coco_train_data = 'coco/coco_train2017/train2017'
coco_valid_data = 'coco/coco_valid2017/val2017'
coco_test_data = 'coco/coco_test2017/test2017'

# data split without fixed stimuli IDs
train_data = 'BOLD5000/bold_train/bold_CSI3_pad.pickle'
valid_data = 'BOLD5000/bold_train/bold_CSI3_pad.pickle'

# stimuli split to fix train and validation sets
train_stimuli_split = 'BOLD5000/bold_roi/stimuli_train.pickle'
valid_stimuli_split = 'BOLD5000/bold_roi/stimuli_valid.pickle'

"""__________________________________training________________________________________________"""

# roi data created with the function 'extract_roi'
bold_roi_data = 'BOLD5000/bold_roi/'
# path for results
save_training_results = 'results/'


"""_________________________________other data parameters____________________________________"""

image_size = 64

subjects = ['CSI1', 'CSI2', 'CSI3', 'CSI4']

rois_max = {'LHEarlyVis': 522,
            'LHLOC': 455,
            'LHOPA': 279,
            'LHRSC': 86,
            'LHPPA': 172,
            'RHEarlyVis': 696,
            'RHLOC': 597,
            'RHOPA': 335,
            'RHRSC': 278,
            'RHPPA': 200}

num_voxels = 3620