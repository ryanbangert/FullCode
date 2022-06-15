# import numpy as np
# import keras
# import tensorflow.keras
# from keras.datasets import cifar10
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization, GlobalAveragePooling2D
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.applications import DenseNet121
# from tensorflow.keras.applications.densenet import preprocess_input
# from sklearn.metrics import accuracy_score

#%% read data
# (x_train_orig, y_train_orig), (x_test, y_test) = cifar10.load_data()

#%% split data
# x_train_1, x_train_2, y_train_1, y_test_2 = train_test_split(x_train_orig, y_train_orig, train_size=0.7)

#%% label or class number 
# num_classes=10

#%% DenseNet121 Architecture
# model_d=DenseNet121(weights='imagenet',include_top=False, input_shape=(128, 128, 3)) 

# x=model_d.output

# x= GlobalAveragePooling2D()(x)
# x= BatchNormalization()(x)
# x= Dropout(0.5)(x)
# x= Dense(1024,activation='relu')(x) 
# x= Dense(512,activation='relu')(x) 
# x= BatchNormalization()(x)
# x= Dropout(0.5)(x)

# preds=Dense(8,activation='softmax')(x) #FC-layer
# model=Model(inputs=base_model.input,outputs=preds)

# for layer in model.layers[:-8]:
#     layer.trainable=False
    
# for layer in model.layers[-8:]:
#     layer.trainable=True
    
# model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# model.fit(x_train_1, y_train_1, batch_size=64, epochs=100)
# test_loss, test_acc = model.evaluate(x_test, y_test_2)
# print("Test Loss", test_loss)
# print("Test Accuracy",test_acc)

#%% AlexNet Architecture
# y_train_1 = to_categorical(y_train_1, 10)

# model = Sequential()

# # 1st Convolutional Layer
# model.add(Conv2D(filters=96, input_shape=(img_height, img_width, channel,), kernel_size=(11,11),\
#  strides=(4,4), padding='same'))
# model.add(Activation('relu'))
# # Pooling 
# model.add(MaxPooling2D(pool_size=(5,5), strides=(2,2), padding='same'))
# # Batch Normalisation before passing it to the next layer
# model.add(BatchNormalization())

# # 2nd Convolutional Layer
# model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
# model.add(Activation('relu'))
# # Pooling
# model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))
# # Batch Normalisation
# model.add(BatchNormalization())

# # 3rd Convolutional Layer
# model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
# model.add(Activation('relu'))
# # Batch Normalisation
# model.add(BatchNormalization())

# # 4th Convolutional Layer
# model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
# model.add(Activation('relu'))
# # Batch Normalisation
# model.add(BatchNormalization())

# # 5th Convolutional Layer
# model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
# model.add(Activation('relu'))
# # Pooling
# model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))
# # Batch Normalisation
# model.add(BatchNormalization())

# # Passing it to a dense layer
# model.add(Flatten())
# # 1st Dense Layer
# model.add(Dense(4096, input_shape=(28*28*3,)))
# model.add(Activation('relu'))
# # Add Dropout to prevent overfitting
# model.add(Dropout(0.4))
# # Batch Normalisation
# model.add(BatchNormalization())

# # 2nd Dense Layer
# model.add(Dense(512))
# model.add(Activation('relu'))
# # Add Dropout
# model.add(Dropout(0.4))
# # Batch Normalisation
# model.add(BatchNormalization())

# # 3rd Dense Layer
# model.add(Dense(256))
# model.add(Activation('relu'))
# # Add Dropout
# model.add(Dropout(0.4))
# # Batch Normalisation
# model.add(BatchNormalization())

# # Output Layer
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

# import tensorflow.keras 

# model.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=tensorflow.keras.optimizers.Adam(), metrics=['accuracy'])
# model.fit(x_train_1, y_train_1, batch_size=64, epochs=100)
# test_loss, test_acc = model.evaluate(x_test, y_test_2)
# print("Test Loss", test_loss)
# print("Test Accuracy",test_acc)

#%% VGG16 Architecture
# model = Sequential()
# model.add(Conv2D(input_shape=(img_height,img_width,channel),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

# model.add(Flatten())
# model.add(Dense(units=4096,activation="relu"))
# model.add(Dense(units=4096,activation="relu"))
# model.add(Dense(units=num_classes, activation="softmax"))

# import tensorflow.keras 

# model.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
# model.fit(train_X, train_Y_one_hot, batch_size=64, epochs=100)
# test_loss, test_acc = model.evaluate(test_X, test_Y_one_hot)
# print("Test Loss", test_loss)
# print("Test Accuracy",test_acc)
#%%% Set Model
# def model_1(x_train, y_train, conv_num, dense_num):
#     input_shape = x_train.shape[1:]
#     # make teacher hot-encoded
#     y_train = to_categorical(y_train, 10)

#     # set model
#     model = Sequential()
#     model.add(Conv2D(conv_num, (3,3), activation='relu', input_shape=input_shape))
#     model.add(Dropout(0.2))
#     model.add(Conv2D(conv_num, (3,3), activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(MaxPooling2D(pool_size=(2,2)))

#     model.add(Conv2D(conv_num * 2, (3,3), activation='relu'))
#     model.add(Conv2D(conv_num * 2, (3,3), activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(MaxPooling2D(pool_size=(2,2)))

#     model.add(Flatten())
#     model.add(Dense(dense_num, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(int(dense_num * 0.6), activation='relu'))
#     model.add(Dense(10, activation='softmax')) # output layer
#     model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=tensorflow.keras.optimizers.Adam(),
#               metrics=['accuracy'])
#     # training
#     history =model.fit(x_train, y_train, batch_size=256, epochs=50, shuffle=True,  validation_split=0.1)
#     return history
# history_1 = model_1(x_train_1, y_train_1, 32, 256)

#%% Predictions for classifiers
# predictions_1 = history_1.model.predict(x_train_2)
# prediction_test = history_1.model.predict(x_test)
#%% KNN Classifier
# from sklearn.neighbors import KNeighborsClassifier
# make models
# knn_2 = KNeighborsClassifier(n_neighbors=2)
# knn_2.fit(predictions_1, y_test_2)
# # predict
# kn_2_pr = knn_2.predict(prediction_test)
#%% SVM Classifier
# from sklearn import svm
# clf = svm.SVC(kernel='linear') # Linear Kernel
# #Train the model using the training sets
# clf.fit(predictions_1, y_test_2)
# #Predict the response for test dataset
# y_pred = clf.predict(prediction_test)
#%% Decision Tree Classifier
# from sklearn.tree import DecisionTreeClassifier
# clf_model = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=3, min_samples_leaf=5)   
# clf_model.fit(predictions_1, y_test_2)
# y_pred = clf_model.predict(prediction_test)
# print("Acc: ",accuracy_score(y_test, y_pred))