# Hyperparameter Tuning with Keras Tuner

This project uses Keras Tuner with TensorFlow/Keras to optimize a neural network for Fashion MNIST image classification. The notebook walks through the full tuning workflow: loading and inspecting the dataset, defining a tunable model factory, extending Keras Tuner with a custom Bayesian optimization tuner that also searches over batch size, and retraining the best discovered configuration with early stopping.

The implementation is centered on a lightweight fully connected architecture for grayscale clothing-image classification. Rather than fixing architecture and training settings upfront, the notebook treats both network design choices and a training-time parameter as search variables, making the experiment a compact example of practical hyperparameter optimization in Keras.

## Objectives

- Create a custom tuner with Keras Tuner.
- Build a hyperparameter tuning experiment for a Keras model.

## Dataset

The project uses the **Fashion MNIST** dataset provided through `tf.keras.datasets.fashion_mnist.load_data()`. Fashion MNIST is a multiclass image-classification benchmark containing low-resolution grayscale images of clothing items.

- Training split: `60,000` images
- Test split: `10,000` images
- Image size: `28 x 28`
- Number of target classes: `10`
- Label space observed in the notebook: `{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}`

The notebook verifies the training tensor shape as `(60000, 28, 28)` and visualizes a sample image with Matplotlib before model construction begins.

## Tech Stack

- TensorFlow `2.19.0`
- `keras-tuner`
- NumPy
- Matplotlib

## Project Workflow

The project follows a clear experiment pipeline:

1. Install `keras-tuner`.
2. Import TensorFlow, Keras Tuner, Matplotlib, and NumPy.
3. Load the Fashion MNIST training and test splits.
4. Inspect dataset shape and label coverage.
5. Visualize a sample Fashion MNIST image.
6. Define a tunable model-building function, `create_model(hp)`.
7. Implement a custom tuner by subclassing `keras_tuner.tuners.BayesianOptimization`.
8. Extend the trial logic so `batch_size` is tuned alongside model hyperparameters.
9. Launch Bayesian hyperparameter search with validation accuracy as the optimization objective.
10. Review the search-space summary and the best trial returned by the tuner.
11. Retrieve the best-performing model configuration.
12. Retrain that model for a longer schedule with early stopping.

## Core Interfaces

Two notebook-defined interfaces drive the project:

```python
create_model(hp)
CustomTuner.run_trial(self, trial, *args, **kwargs)
```

- `create_model(hp)` constructs and compiles a `tf.keras.Sequential` classifier. When a hyperparameter object is supplied, the model structure and optimizer learning rate are drawn from the search space.
- `CustomTuner.run_trial(...)` overrides Keras Tuner's default trial execution so that `batch_size` becomes part of the hyperparameter search, even though it is not a layer-level model attribute.

## Model Architecture

The model is created through `create_model(hp)`. The function defines a default configuration and then replaces those defaults with tuned values whenever a Keras Tuner hyperparameter object is passed in.

### Default configuration when `hp` is `None`

- Hidden layers: `1`
- Units per hidden layer: `8`
- Dropout rate: `0.1`
- Learning rate: `0.01`

This default model produces a network with `6,370` trainable parameters, based on the notebook's `create_model(None).summary()` output.

### Input processing

The input pipeline is embedded directly into the model:

1. `Flatten(input_shape=(28, 28))` reshapes each image into a 784-dimensional vector.
2. `Lambda(lambda x: x / 255.0)` performs in-model normalization by scaling pixel values to the `[0, 1]` range.

### Hidden and output layers

- A loop adds between `1` and `3` hidden layers, depending on the selected hyperparameter configuration.
- Each hidden layer is a `Dense(..., activation='relu')` layer.
- Each hidden layer is immediately followed by `Dropout(...)`.
- The classifier head is `Dense(10, activation='softmax')`.

### Compilation settings

The model is compiled with:

- Loss: `sparse_categorical_crossentropy`
- Optimizer: `tf.keras.optimizers.Adam(learning_rate=learning_rate)`
- Metric: `accuracy`

## Hyperparameter Search Strategy

The notebook uses Bayesian optimization to search across both architectural and training-time parameters.

### Search space

| Hyperparameter | Type | Values / Range | Role |
| --- | --- | --- | --- |
| `num_hidden_layers` | Choice | `1`, `2`, `3` | Controls network depth |
| `num_units` | Choice | `8`, `16`, `32` | Controls width of each hidden layer |
| `dropout_rate` | Float | `0.1` to `0.5` | Controls regularization strength |
| `learning_rate` | Float | `0.0001` to `0.01` | Controls Adam optimizer step size |
| `batch_size` | Int | `32` to `128`, step `32` | Controls samples processed per optimization step |

The first four parameters are consumed inside `create_model(hp)`. The fifth parameter, `batch_size`, is introduced by overriding `run_trial`.

### Tuner configuration

The tuner is instantiated as a subclass of `keras_tuner.tuners.BayesianOptimization` with the following configuration:

- Objective: `val_accuracy`
- Maximum trials: `20`
- Search log directory: `logs`
- Project name: `fashion_mnist`
- Overwrite existing search state: `True`
- Search epochs per trial: `5`
- Validation data: `(x_test, y_test)`

This setup instructs Keras Tuner to explore candidate configurations, score them on validation accuracy, and record results under `logs/fashion_mnist` during execution.

## Custom Tuner Implementation

The custom tuner is an important part of the project because it expands the tuning problem beyond the model factory itself. In standard usage, Keras Tuner typically searches over values exposed inside the model-building function. Here, the notebook overrides `run_trial` so that `batch_size` is sampled from:

- minimum: `32`
- maximum: `128`
- step: `32`

That value is then injected into `kwargs['batch_size']` before delegating back to the parent `BayesianOptimization` implementation. This makes the experiment more realistic because optimization now spans:

- model depth
- model width
- dropout strength
- optimizer learning rate
- mini-batch size

In other words, the notebook demonstrates tuning across both **model-building hyperparameters** and **training-process hyperparameters**.

## Results

The notebook reports results in two distinct stages: the best trial discovered during the short Bayesian search, and the performance observed when the selected model is retrained for longer.

### Best tuning trial

The best trial reported by `tuner.results_summary(1)` is:

| Parameter | Best value |
| --- | --- |
| `num_hidden_layers` | `2` |
| `num_units` | `32` |
| `dropout_rate` | `0.1184154753605729` |
| `learning_rate` | `0.0006264272379225366` |
| `batch_size` | `128` |
| Search objective score | `val_accuracy = 0.8504999876022339` |

This score corresponds to the tuner stage, where each candidate configuration is evaluated for `5` epochs.

### Best model selected from tuning

After the search, the notebook retrieves the top model with:

```python
model = tuner.get_best_models(num_models=1)[0]
```

The retrieved model summary shows a two-hidden-layer dense network with `32` units per hidden layer and `26,506` trainable parameters.

### Retraining outcome

The selected model is then retrained with:

- up to `20` epochs
- `batch_size=128`
- `EarlyStopping(monitor='val_accuracy', patience=3)`

During this longer training stage, the notebook output shows validation accuracy improving beyond the original tuning score and reaching `0.8730`.

### Evaluation note

The notebook uses the provided Fashion MNIST test split as `validation_data` during both the tuning phase and the retraining phase. Because that split is reused for model selection and monitoring, the reported validation accuracies should be interpreted as experiment validation metrics within the notebook workflow rather than as an untouched final benchmark.

## Technical Takeaways

- Bayesian search moved the model away from the default lightweight configuration toward a larger two-layer network with wider hidden representations and a tuned learning rate.
- Normalization is handled inside the network through a `Lambda` layer, keeping preprocessing embedded in the model definition itself.
- The project demonstrates that Keras Tuner can optimize more than architecture alone when trial execution is customized.
- Even in a compact notebook, it is possible to combine automated search with a follow-up retraining stage to better evaluate the selected configuration.
