from keras.models import Sequential
from keras.layers import Dense


# def create_baseline():
#     model = Sequential()
#     model.add(Dense(30, input_dim=513, kernel_initializer='normal', activation='relu'))
#     #model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
#     model.add(Dense(3, kernel_initializer='normal', activation='softmax'))
#     # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model

def create_baseline():
    model = Sequential()
    model.add(Dense(100, input_dim=513, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, input_dim=100, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(3, kernel_initializer='normal', activation='softmax'))
    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
