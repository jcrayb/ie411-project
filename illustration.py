from PIL import Image
import numpy as np

im = Image.open('./tower.png').convert('L')

i1 = np.array(im)

h1, w1 = i1.shape

source = 49

for i in range(h1):
    for j in range(w1):
        i1[i, j] = 0 if source-i>=j or j>=source+i else i1[i, j]

new_img = Image.fromarray(i1)
new_img.save('reduced1.png')

i2 = np.array(im)

h2, w2 = i2.shape

source = 49

for i in range(h2):
    for j in range(w2):
        i2[i, j] = 0 if source-i>=j or j>=source+i or j>=100 else i1[i, j]

new_img = Image.fromarray(i2)
new_img.save('reduced2.png')