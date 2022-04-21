import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os


# load images
images = []
dir = r'workplace\classification_for_vision\image_data\standart'
base_dir = os.path.dirname(dir)
filenames = os.listdir(dir)


for x in filenames:
    path = os.path.join(dir, x)
    img = Image.open(path)
    images.append(np.asarray(img))



# Rot_180
rot180_dir = os.path.join(base_dir, 'rotated_180')
aug_180 = iaa.Rot90(2)
images_180 = aug_180(images=images)
# #save
for x, name in zip(images_180, filenames):
    pil_image = Image.fromarray(x)
    pil_image.save(os.path.join(rot180_dir, name))

# Rot90 right(image is falling right)
rot90_right_dir = os.path.join(base_dir, 'rotated_90_right')
aug_right = iaa.Rot90(1, keep_size=False)
images_right = aug_right(images=images)
# #save
for x, name in zip(images_right, filenames):
    pil_image = Image.fromarray(x)
    pil_image.save(os.path.join(rot90_right_dir, name))


# Rot90 left ( image is falling left )
rot90_left_dir = os.path.join(base_dir, 'rotated_90_left')
aug_left = iaa.Rot90(3, keep_size=False)
images_left = aug_left(images=images)
# #save
for x, name in zip(images_left, filenames):
    pil_image = Image.fromarray(x)
    pil_image.save(os.path.join(rot90_left_dir, name))


# AUGMENT and increase datasize with rotation:
seq = iaa.Sequential([iaa.Affine(rotate=(-45, 45), order=0, mode='edge')])

# #standart
images_aug_standart = seq(images=images)
#save
for x, name in zip(images_aug_standart, filenames):
    pil_image = Image.fromarray(x)
    pil_image.save(os.path.join(dir, name.replace('.', '_rotated.')))


# #180
images_aug_180 = seq(images=images_180)
#save
for x, name in zip(images_aug_180, filenames):
    pil_image = Image.fromarray(x)
    pil_image.save(os.path.join(rot180_dir, name.replace('.', '_rotated.')))

# #90 rotated right
images_aug_rot90right = seq(images=images_right)
#save
for x, name in zip(images_aug_rot90right, filenames):
    pil_image = Image.fromarray(x)
    pil_image.save(os.path.join(rot90_right_dir, name.replace('.', '_rotated.')))

# #90 rotated left
images_aug_rot90left = seq(images=images_left)
#save
for x, name in zip(images_aug_rot90left, filenames):
    pil_image = Image.fromarray(x)
    pil_image.save(os.path.join(rot90_left_dir, name.replace('.', '_rotated.')))






'''
#cv2.imshow('test', images_aug[5])
#cv2.waitKey(0)
fig = plt.figure(figsize=(8, 4))
plt.imshow(images[1])

images_aug = images_aug_standart
f, axarr = plt.subplots(4,2)
axarr[0,0].imshow(images_aug[1])
axarr[0,1].imshow(images_aug[2])
axarr[1,0].imshow(images_aug[3])
axarr[1,1].imshow(images_aug[4])
axarr[2,0].imshow(images_aug[6])
axarr[2,1].imshow(images_aug[7])
axarr[3,0].imshow(images_aug[8])
axarr[3,1].imshow(images_aug[9])
'''