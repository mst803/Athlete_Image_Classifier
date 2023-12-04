import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from tqdm import tqdm
import imgaug.augmenters as iaa

messi_images=os.listdir("lionel_messi")
sharapova_images=os.listdir("maria_sharapova")
federer_images=os.listdir("roger_federer")
williams_images=os.listdir("serena_williams")
kohli_images=os.listdir("virat_kohli")


dataset=[]
label=[]
img_siz=(128,128)
# Define augmentation pipeline
seq = iaa.Sequential([
    iaa.Fliplr(0.2),  # Horizontal flips
    iaa.Crop(percent=(0, 0.1)),  # Random crops
    iaa.GaussianBlur(sigma=(0, 0.5)),  # Gaussian blur
    iaa.Affine(rotate=(-15, 15)),  # Random rotations
    # Add more augmentation techniques as needed
])

n_a=3

for i , image_name in tqdm(enumerate(messi_images),desc="messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(r"lionel_messi\{}".format(image_name))
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(0)
        # Apply augmentation and append augmented images to dataset and label
        for j in range(n_a):
            augmented_images = seq(images=[np.array(image)])
            dataset.extend(augmented_images)
            label.extend([0] * len(augmented_images))
        
for i , image_name in tqdm(enumerate(sharapova_images),desc="sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(r"maria_sharapova\{}".format(image_name))
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(1)
        # Apply augmentation and append augmented images to dataset and label
        for j in range(n_a):
            augmented_images = seq(images=[np.array(image)])
            dataset.extend(augmented_images)
            label.extend([1] * len(augmented_images))
        
for i , image_name in tqdm(enumerate(federer_images),desc="federer"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(r"roger_federer\{}".format(image_name))
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(2)
        # Apply augmentation and append augmented images to dataset and label
        for j in range(n_a):
            augmented_images = seq(images=[np.array(image)])
            dataset.extend(augmented_images)
            label.extend([2] * len(augmented_images))
        
for i , image_name in tqdm(enumerate(williams_images),"williams"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(r"serena_williams\{}".format(image_name))
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(3)
        # Apply augmentation and append augmented images to dataset and label
        for j in range(n_a):
            augmented_images = seq(images=[np.array(image)])
            dataset.extend(augmented_images)
            label.extend([3] * len(augmented_images))

for i , image_name in tqdm(enumerate(kohli_images),"kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(r"virat_kohli\{}".format(image_name))
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(4)
        # Apply augmentation and append augmented images to dataset and label
        for j in range(n_a):
            augmented_images = seq(images=[np.array(image)])
            dataset.extend(augmented_images)
            label.extend([4] * len(augmented_images))
        
        
dataset=np.array(dataset)
label = np.array(label)

#for i in range(300):
    #cv2.imshow('Image', dataset[i])
    #cv2.waitKey(1000)
    #cv2.destroyAllWindows()
    #print(label[i])

print(len(dataset))
print(len(label))


X_train=[]
y_train=[]
X_valid=[]
y_valid=[]
X_test=[]
y_test=[]

for i in range(len(dataset)):
    if i%5==0:
        X_test.append(dataset[i])
        y_test.append(label[i])
    elif i%4==0:
        X_valid.append(dataset[i])
        y_valid.append(label[i])
    else:
        X_train.append(dataset[i])
        y_train.append(label[i])
        
# x_train=x_train.astype('float')/255
# x_test=x_test.astype('float')/255 

# Same step above is implemented using tensorflow functions.

X_train = tf.keras.utils.normalize(np.array(X_train), axis=-1)
X_test = tf.keras.utils.normalize(np.array(X_test), axis=-1)
X_valid = tf.keras.utils.normalize(np.array(X_valid), axis=-1)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_valid = np.array(X_valid)
y_valid = np.array(y_valid)
X_test = np.array(X_test)
y_test = np.array(y_test)

print("--------------------------------------\n")


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

print("--------------------------------------\n")
model.summary()
print("--------------------------------------\n")

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)

print("--------------------------------------\n")
print("Training Started.\n")
history=model.fit(X_train,y_train,epochs=100,validation_data=(X_valid,y_valid),callbacks=[early_stopping],batch_size =32)

print("Training Finished.\n")
print("--------------------------------------\n")

print("--------------------------------------\n")

print("Model Evalutaion Phase.\n")
loss,accuracy=model.evaluate(X_test,y_test)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")


# Plot and save accuracy
plt.plot(history.epoch,history.history['accuracy'], label='accuracy')
plt.plot(history.epoch,history.history['val_accuracy'], label = 'val_accuracy')
plt.plot(history.epoch,history.history['loss'], label='loss')
plt.plot(history.epoch,history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig("accuracy_plot.png")


def make_prediction(img,model):
    img=cv2.imread(img)
    img=Image.fromarray(img)
    img=img.resize((128,128))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    res = model.predict(input_img,verbose=0)
    if res==0:
        print("Messi")
    elif res==1:
        print("sharapova")
    elif res==2:
        print("federer")
    elif res==3:
        print("williams")
    elif res==4:
        print("kohli")
    else:
        "Error....."

model.save('cnn_celeb.keras')