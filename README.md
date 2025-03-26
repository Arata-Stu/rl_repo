# RL

## setup
```shell
python3.11 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## run

### 1. collect data
```shell
python3 collect_data.py output_dir=/path/to/output
```

### 2. train VAE
```shell
SHAPE=[3,64,64]
python3 train_vae.py \
vae.input_shape=${SHAPE} \
dataset.name=img dataset.root=/path/to/dataset \
logger.type=tensorboard 
```

### 3. train_actor
```shell
SHAPE="[3, 64, 64]"
LATENT=64
python3 train_actor.py \
vae.input_shape=${SHAPE} \
vae.latent_dim=${LATENT}

```