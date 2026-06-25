import tensorflow as tf
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    DEVICE = '/GPU:0'
    print(f"✅ Using GPU: {gpus[0].name}")
else:
    DEVICE = '/CPU:0'
    print("⚠️  Using CPU")

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(BASE_DIR, 'models')
METRICS_DIR  = os.path.join(BASE_DIR, 'outputs', 'metrics')
PLOTS_DIR    = os.path.join(BASE_DIR, 'outputs', 'plots')
CKPT_DIR     = os.path.join(BASE_DIR, 'checkpoints')

IMAGE_SIZE     = 28
CHANNELS       = 1
LATENT_DIM     = 100
NUM_CLASSES    = 10
CLASS_NAMES    = ['T-shirt','Trouser','Pullover','Dress','Coat',
                   'Sandal','Shirt','Sneaker','Bag','Ankle boot']

GEN_LR         = 0.0002
DISC_LR        = 0.0002
BETA_1         = 0.5
BATCH_SIZE     = 128       # reduce to 64 if GPU OOM (RTX 2050)
EPOCHS         = 50
NUM_EXAMPLES   = 16         # for sample grid
SEED           = 42

tf.random.set_seed(SEED)
