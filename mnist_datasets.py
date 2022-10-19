import tensorflow as tf 
(x,y),(tx,ty)=tf.keras.datasets.mnist.load_data()

xt=x.reshape(-1,28*28)/255.0
tx=tx.reshape(-1,28*28)/255.0
model=tf.keras.Sequential()

model.add(tf.keras.layers.Dense(10,input_shape=(784,),activation='sigmoid',name='fc1'))
model.add(tf.keras.layers.Dense(100,activation='relu',name='fc2'))
model.add(tf.keras.layers.Dense(10,activation='sigmoid',name='output'))


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#print(model.summary())
model.fit(xt,y,epochs=10)

result= model.evaluate(tx,ty)

print('loss = ' + str(result[0]))
print('accuracy = ' + str(result[1]))
