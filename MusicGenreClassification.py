from utils import load_data, run_test, plot, draw_confusion_matrix
from MusicRecNet import MusicRecNet

DATA_PATH = "saved_dataset/"


"""## Experiments on MusicRecNet

### MusicRecNet Model Configurations
"""

# basic MusicRecNet
conv_configs_basic_1 = [
    {"num_filters": 32, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": False, "basic": True}, 
    {"num_filters": 64, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": False}, 
    {"num_filters": 128, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": False}
]

# best MusicRecNet with more layers
conv_configs_basic_2 = [
    {"num_filters": 32, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": False, "basic": True}, 
    {"num_filters": 64, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": False}, 
    {"num_filters": 128, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": False},
    {"num_filters": 256, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": False}
]

"""### Experiment 1: 32, 64, 128 filters on 128_128_6seg"""

data_name = "128_128_6seg"
conv_configs = conv_configs_basic_1
epochs = 150

model, hist, z, X_test, y_test = run_test(data_name=data_name,
                                          conv_configs=conv_configs,
                                          colab_folder=DATA_PATH,
                                          epochs=epochs,
                                          summary=True,
                                          verbose=1)

plot(hist, path="./", name=None, save=False)

draw_confusion_matrix(X=X_test, y_true=y_test, model=model, labels=z, path="./", name=None, save=False)

"""### Experiment 2: 32, 64, 128 filters on 128_128_10seg"""

data_name = "128_128_10seg"
conv_configs = conv_configs_basic_1
epochs = 150

model, hist, z, X_test, y_test = run_test(data_name=data_name,
                                          conv_configs=conv_configs,
                                          colab_folder=DATA_PATH,
                                          epochs=epochs,
                                          summary=True,
                                          verbose=1)

plot(hist, path="./", name=None, save=False)

draw_confusion_matrix(X=X_test, y_true=y_test, model=model, labels=z, path="./", name=None, save=False)

"""### Experiment 3: 32, 64, 128 filters on 128_128_15seg"""

# Experiment 3: 32, 64, 128 filters on 128_128_15seg
data_name = "128_128_15seg"
conv_configs = conv_configs_basic_1
epochs = 200

model, hist, z, X_test, y_test = run_test(data_name=data_name,
                                          conv_configs=conv_configs,
                                          colab_folder=DATA_PATH,
                                          epochs=epochs,
                                          summary=True,
                                          verbose=1)

plot(hist, path="./", name=None, save=False)

draw_confusion_matrix(X=X_test, y_true=y_test, model=model, labels=z, path="./", name=None, save=False)

"""### Experiment 4: 32, 64, 128 filters on 128_128_16seg"""

# Experiment 4: 32, 64, 128 filters on 128_128_16seg
data_name = "128_128_15seg"
conv_configs = conv_configs_basic_2
epochs = 150

model, hist, z, X_test, y_test = run_test(data_name=data_name,
                                          conv_configs=conv_configs,
                                          colab_folder=DATA_PATH,
                                          epochs=epochs,
                                          summary=True,
                                          verbose=1)

plot(hist, path="./", name=None, save=False)

draw_confusion_matrix(X=X_test, y_true=y_test, model=model, labels=z, path="./", name=None, save=False)

"""## Experiments on ResNet-MusicRecNet

### RecNet-MusicRecNet Model Configurations
"""

# Setting 1: 32, 64 (Res Block), 128 (Res Block) filters
# ResNet Block
conv_configs_res_1 = [
    {"num_filters": 32, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": False, "basic": False},
    {"num_filters": 64, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": True, "batch_normal": True, "noise": None},
    {"num_filters": 128, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": True, "batch_normal": True, "noise": None}
]
# DN-ResNet Block
conv_configs_dnres_1 = [
    {"num_filters": 32, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": False, "basic": False},
    {"num_filters": 64, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": True, "batch_normal": False, "noise": None},
    {"num_filters": 128, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": True, "batch_normal": False, "noise": None}
]


# Setting 2: 32, 64 (Res Block), 128 filters
# ResNet Block
conv_configs_res_2 = [
    {"num_filters": 32, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": False, "basic": False},
    {"num_filters": 64, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": True, "batch_normal": True, "noise": None},
    {"num_filters": 128, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": False}
]
# DN-ResNet Block
conv_configs_dnres_2 = [
    {"num_filters": 32, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": False, "basic": False},
    {"num_filters": 64, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": True, "batch_normal": False, "noise": None},
    {"num_filters": 128, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": False}
]


# Setting 3: 32, 64 (Res Block), 64 filters
# ResNet Block
conv_configs_res_3 = [
    {"num_filters": 32, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": False, "basic": False},
    {"num_filters": 64, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": True, "batch_normal": True, "noise": None},
    {"num_filters": 64, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": False}
]
# DN-ResNet Block
conv_configs_dnres_3 = [
    {"num_filters": 32, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": False, "basic": False},
    {"num_filters": 64, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": True, "batch_normal": False, "noise": None},
    {"num_filters": 64, "kernel_size": (3, 3), "activation": "relu", "pool_size": (2, 2), "resNet": False}
]

"""### Setting 1: 32, 64 (Res Block), 128 (Res Block) filters

#### ResNet Block
"""

data_name = "128_128_15seg"
conv_configs = conv_configs_res_1
epochs = 20

model, hist, z, X_test, y_test = run_test(data_name=data_name,
                                          conv_configs=conv_configs,
                                          colab_folder=DATA_PATH,
                                          epochs=epochs,
                                          summary=True,
                                          verbose=1)

plot(hist, path="./", name=None, save=False)

draw_confusion_matrix(X=X_test, y_true=y_test, model=model, labels=z, path="./", name=None, save=False)

"""#### DN-ResNet Block"""

data_name = "128_128_15seg"
conv_configs = conv_configs_dnres_1
epochs = 20

model, hist, z, X_test, y_test = run_test(data_name=data_name,
                                          conv_configs=conv_configs,
                                          colab_folder=DATA_PATH,
                                          epochs=epochs,
                                          summary=True,
                                          verbose=1)

plot(hist, path="./", name=None, save=False)

draw_confusion_matrix(X=X_test, y_true=y_test, model=model, labels=z, path="./", name=None, save=False)

"""### Setting 2: 32, 64 (Res Block), 128 filters

#### ResNet Block
"""

data_name = "128_128_15seg"
conv_configs = conv_configs_res_2
epochs = 20

model, hist, z, X_test, y_test = run_test(data_name=data_name,
                                          conv_configs=conv_configs,
                                          colab_folder=DATA_PATH,
                                          epochs=epochs,
                                          summary=True,
                                          verbose=1)

plot(hist, path="./", name=None, save=False)

draw_confusion_matrix(X=X_test, y_true=y_test, model=model, labels=z, path="./", name=None, save=False)

"""#### DN-ResNet Block"""

data_name = "128_128_15seg"
conv_configs = conv_configs_dnres_2
epochs = 20

model, hist, z, X_test, y_test = run_test(data_name=data_name,
                                          conv_configs=conv_configs,
                                          colab_folder=DATA_PATH,
                                          epochs=epochs,
                                          summary=True,
                                          verbose=1)

plot(hist, path="./", name=None, save=False)

draw_confusion_matrix(X=X_test, y_true=y_test, model=model, labels=z, path="./", name=None, save=False)

"""### Setting 3: 32, 64 (Res Block), 64 filters

#### ResNet Block
"""

data_name = "128_128_15seg"
conv_configs = conv_configs_res_3
epochs = 20

model, hist, z, X_test, y_test = run_test(data_name=data_name,
                                          conv_configs=conv_configs,
                                          colab_folder=DATA_PATH,
                                          epochs=epochs,
                                          summary=True,
                                          verbose=1)

plot(hist, path="./", name=None, save=False)

draw_confusion_matrix(X=X_test, y_true=y_test, model=model, labels=z, path="./", name=None, save=False)

"""#### DN-ResNet Block"""

data_name = "128_128_15seg"
conv_configs = conv_configs_dnres_3
epochs = 20

model, hist, z, X_test, y_test = run_test(data_name=data_name,
                                          conv_configs=conv_configs,
                                          colab_folder=DATA_PATH,
                                          epochs=epochs,
                                          summary=True,
                                          verbose=1)

plot(hist, path="./", name=None, save=False)

draw_confusion_matrix(X=X_test, y_true=y_test, model=model, labels=z, path="./", name=None, save=False)