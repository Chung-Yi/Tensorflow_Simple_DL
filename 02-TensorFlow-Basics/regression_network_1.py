import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def main():
    x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
    y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
    
    # plt.plot(x_data, y_label, '*')
    # plt.show()
    # print(np.random.rand(2))
    m = tf.Variable(0.59)
    b = tf.Variable(0.43)

    error = 0
    for x, y in zip(x_data, y_label):
        y_hat = m * x + b
        error += (y - y_hat) ** 2
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

    train = optimizer.minimize(error)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        training_steps = 1000
        for i in range(training_steps):
            sess.run(train)
        final_slop, final_intercept = sess.run([m, b])
    

    x_test = np.linspace(-1, 11, 10)
    y_pred_plot = final_slop * x_test + final_intercept
    plt.plot(x_test, y_pred_plot, 'r')
    plt.plot(x_data, y_label, '*')
    plt.show()


if __name__ == '__main__':
    main()