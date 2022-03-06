import tensorflow as tf
import tensorflow_datasets as tfds

mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)
# with_info=True provides tuple containing information about the version, features, number of samples. I stored these info to in the mnist_info
# as_supervised=True loads the dataset as a 2-tuple structure (input, target) to separate them.

mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']
# Here I extracted the training and testing datasets with the built-in references
num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)
# I defined the number of validation samples (10%) as a % of the training samples and cast the number as an integer.
num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)


# I stored the number of test samples in a dedicated variable.

def scale(image, label):
    image = tf.cast(image, tf.float32)
    # The value must be a float since the possible values for the inputs are 0 to 255 (256 different shades of grey)
    image /= 255.
    # Returns a value between 0 and 1.
    return image, label


# The function 'scale' takes an MNIST image and its label. It prefers inputs between 0 and 1.

scaled_train_and_validation_data = mnist_train.map(scale)

test_data = mnist_test.map(scale)
# The method .map() applies a custom transformation to the given dataset.
# I scaled them all, so they have the same magnitude.
# There was no need to shuffle the test data because I didn't train my model using it.
# I decided that a single Batch would be equal to the size of the test data to hasten the process.

BUFFER_SIZE = 10000
# I can't shuffle the whole dataset in one go because it can't fit in memory, so I set this BUFFER_SIZE parameter for this huge dataset.

shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)
# I shuffled the train and validation data after setting the BUFFER_SIZE.
validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
# Here I used the .take() method to take my validation data equal to 10% of the training set.
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)
# The train_data is everything else, so I skipped as many samples as there are in the validation dataset.

BATCH_SIZE = 150
# This is my hyperparameter

train_data = train_data.batch(BATCH_SIZE)
# Here I took the advantage to batch the train data to be able to iterate over the different batches.
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)
# I batched the train and test datasets here.
validation_inputs, validation_targets = next(iter(validation_data))

input_size = 784
output_size = 10
# Use same hidden layer size for both hidden layers. Not a necessity.
# There is a convenient method 'Flatten' that simply takes our 28x28x1 tensor and orders it into a (None,) or (28x28x1,) = (784,) vector
hidden_layer_size = 2500

model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    # This is the first layer (the input layer)
    # each observation is 28x28x1 pixels, therefore it is a tensor of rank 3
    # I used the 'flatten' to actually create a feed forward neural network
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    # 2nd hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    # 3rd hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    # 4th hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    # 5th hidden layer
    # This code outputs: activation(dot(input, weight) + bias)
    tf.keras.layers.Dense(output_size, activation='softmax')  # output layer
    # the final layer is no different, except that I activated it using the Softmax method.

])

custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

NUM_EPOCHS = 10

model.fit(train_data,
          epochs=NUM_EPOCHS,
          validation_data=(validation_inputs, validation_targets),
          validation_steps=10, verbose=2)

test_loss, test_accuracy = model.evaluate(test_data)
print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy * 100.))
# Better formatting
