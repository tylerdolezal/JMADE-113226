from tensorflow import keras
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np


# Generate or load your training data
# X holds the energy, force (Fx, Fy, Fz per atom), and stress of the sim cells
# y holds the elastic properties of the sim cells (from elastool)
def neural_network():

    X, y = np.load("x-matrix.npy"), np.load("y-matrix.npy")
    Xscaler = StandardScaler()
    yscaler = StandardScaler()
    X, y = Xscaler.fit_transform(X), yscaler.fit_transform(y)
    print(X.shape,y.shape)
    # Split the data into training and testing sets
    # Create the CNN model
    model = keras.Sequential()

    model.add(Dense(800, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(800, activation='relu'))
    model.add(Dense(300, activation='tanh'))
    model.add(Dense(y.shape[1], activation='linear'))  # Output layer with 9 indices

    # Compile the model
    model.compile(optimizer='adam', loss=Huber())

    # Train the model
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss for early stopping
    patience=5,  # Number of epochs with no improvement after which training will be stopped
    verbose=1,  # Verbosity mode (1: update messages)
    restore_best_weights=True  # Restore the weights of the best epoch
    )

    for train_index, test_index in kf.split(X):
        # Split the data into training and test sets for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the model with EarlyStopping
        history = model.fit(
            X_train, y_train,
            epochs=100,  # Maximum number of epochs
            batch_size=64,
            validation_data=(X_test, y_test),  # Validation data for early stopping
            callbacks=[early_stopping], # Pass the EarlyStopping callback to the fit method
            verbose=0
            )


    predictions = yscaler.inverse_transform(model.predict(X_test))
    predict_y = yscaler.inverse_transform(model.predict(X_train))

    y_train = yscaler.inverse_transform(y_train)
    y_test = yscaler.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(y_train, predict_y))
    print("Train root mean squared:", rmse)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print("Test root mean squared:", rmse)

    model.save("model_for_constants")

    # save the transformation used for model training
    from joblib import dump

    dump(Xscaler,"Xscaler.joblib")
    dump(yscaler, "yscaler.joblib")

neural_network()
