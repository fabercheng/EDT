import os
import tensorflow as tf
from rdkit import Chem
from bs4 import BeautifulSoup

# TensorFlow CUDA 环境配置
def configure_tensorflow_cuda():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(f"Error configuring TensorFlow and CUDA: {e}")

def configure_rdkit():
    pass
def configure_bs4():
    pass

def configure_all():
    configure_tensorflow_cuda()
    configure_rdkit()
    configure_bs4()

if __name__ != "__main__":
    configure_all()
