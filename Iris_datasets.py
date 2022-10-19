
import tensorflow as tf 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
df=load_iris()

X=df.data
Y=df.target

(x_train,x_test,y_train,y_test)=train_test_split(X,Y,test_size=0.2)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(10,input_shape=(4,),activation='relu',name='fc1'))
model.add(tf.keras.layers.Dense(50,activation='relu',name='fc3'))
model.add(tf.keras.layers.Dense(3,activation='softmax',name='output'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


print(model.summary())

model.fit(x_train,y_train,epochs=200)

model.save('Iris')

result=model.evaluate(x_test,y_test)
print('loss  = '  + str(result[0]))
print('accuracy = ' + str(result[1]))
