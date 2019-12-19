import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def tf_plh_var(x_data, y_true):
    x_df = pd.DataFrame(data=x_data, columns=['X_Data'])
    y_df = pd.DataFrame(data=y_true, columns=['Y'])
    
    data = pd.concat([x_df, y_df], axis=1)
    data.sample(n=250).plot(kind='scatter', x='X_Data', y='Y')
    plt.show()

    batch_size = 8
    
    m = tf.Variable(0.5)
    b = tf.Variable(1.0)

    xph = tf.placeholder(tf.float32, batch_size)
    yph = tf.placeholder(tf.float32, batch_size)

    y_model = m * xph + b

    error = tf.reduce_sum(tf.square(yph - y_model))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train = optimizer.minimize(error)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        batches = 1000
        for i in range(batches):
            rand_id = np.random.randint(len(x_data), size=batch_size)
            sess.run(train, feed_dict={xph: x_data[rand_id], yph: y_true[rand_id]})
        model_m, model_b = sess.run([m, b])

        print(model_m, model_b)

        y_hat = model_m * x_data + model_b
        data.sample(n=250).plot(kind='scatter', x='X_Data', y='Y')
        plt.plot(x_data, y_hat, 'r')
        plt.show()


def tf_estimator(x_data, y_true):

    x_df = pd.DataFrame(data=x_data, columns=['X_Data'])
    y_df = pd.DataFrame(data=y_true, columns=['Y'])
    
    data = pd.concat([x_df, y_df], axis=1)

    feat_cols = [tf.feature_column.numeric_column(key='x', shape=[1])] # number of feat_cols is not always equal 1, so using list
    estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)
    
    x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true, test_size=0.3, random_state=101)
    
    input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=None, shuffle=True)
    train_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=1000, shuffle=False)
    eval_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_eval}, y_eval, batch_size=8, num_epochs=1000, shuffle=False)

    estimator.train(input_fn=input_func, steps=1000)
    train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)
    eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)
    
    print('TRAINING DATA METRICS')
    print(train_metrics)

    print('EVAL METRICS')
    print(eval_metrics)

    brand_new_data = np.linspace(0, 10, 10)
    input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x': brand_new_data}, shuffle=False)
    # print(list(estimator.predict(input_fn=input_fn_predict)))

    predictions = []

    for pred in estimator.predict(input_fn=input_fn_predict):
        predictions.append(pred['predictions'])
    
    print(predictions)

    data.sample(n=250).plot(kind='scatter', x='X_Data', y='Y' )
    plt.plot(brand_new_data, predictions, 'ro')
    plt.show()


def main():
    x_data = np.linspace(0.0, 10.0, 1000000)
    noise = np.random.randn(len(x_data))

    
    y_true = 0.8 * x_data + 5 + noise


    # using placeholder and variable
    # tf_plh_var(x_data, y_true)
    

    # tf estimator
    tf_estimator(x_data, y_true)




if __name__ == '__main__':
    main()