import torch
import os
import yaml
import wandb

from wandb.wandb_run import Run
from models import subj
from models.lightning import DataModule, BaseModel

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

os.environ["WANDB_CONSOLE"] = "off"  # Needed to avoid "ValueError: signal only works in main thread of the main interpreter".

# reduces mat mul precision (for performance)
torch.set_float32_matmul_precision('high')

### Variables ###
# reads setting`s yaml

with open(os.path.join(os.path.dirname(__file__), "settings.yaml"), "r") as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)
    # global variables
    SWEEP_ID = settings["sweep_id"]
    MODEL_TASK = settings["task"]
    MODEL_VERSION = settings["nn_model"]
    # training hyperparams
    MAX_EPOCHS = settings["max_epochs"]
    LR_PATIENCE = settings["learning_rate_patience"]
    #GRADIENT_CLIPPING = settings["gradient_clipping"]

# reads sweep configs yaml
with open(os.path.join(os.path.dirname(__file__),'sweep_config.yaml')) as file:
    SWEEP_CONFIGS = yaml.load(file, Loader=yaml.FullLoader)

networks_choices = {"subj_classifier": {"subj_classifier": subj.subj_classifier},
                      "cc_classifier": {}}
MODEL_NETWORK = networks_choices[MODEL_TASK][MODEL_VERSION]

loss_choices = {"binary_cross_entropy":torch.nn.BCEWithLogitsLoss()}

dataset_id = {"subj_classifier": 1,
              }

# defines path dir
CHECKPOINT_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", f"{MODEL_TASK}", f"{MODEL_VERSION}")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "embeddings")

# creates directories
os.makedirs(CHECKPOINT_SAVE_PATH, exist_ok =True)

### Main ###
def main():
    # starts wandb
    with wandb.init(config=SWEEP_CONFIGS) as run:
        assert isinstance(run, Run)
        # initialize logger
        logger = WandbLogger(project="LLMs", experiment=run)
        # gets sweep configs
        configs = run.config.as_dict()
        # checkpoint callback setting
        checkpoint_callback = ModelCheckpoint(dirpath=CHECKPOINT_SAVE_PATH, filename= run.name, save_top_k=1, monitor='Loss/Val', mode='min', enable_version_counter=False, save_last=False, save_weights_only=True)
        # load data module
        data_module = DataModule(datasets_root=EMBEDDINGS_PATH, batch_size= configs["batch_size"], dataset_id=dataset_id[MODEL_TASK], num_workers=6 )
        model = BaseModel(model=MODEL_NETWORK, loss_function=loss_choices[configs["loss_function"]], batch_size=configs["batch_size"], learning_rate=configs["lr"], learning_rate_patience=LR_PATIENCE, dataset_id=dataset_id[MODEL_TASK])
        # train the model
        trainer = Trainer(
                        logger= logger,
                        accelerator="gpu",
                        max_epochs=MAX_EPOCHS,
                        callbacks= [checkpoint_callback,
                                    LearningRateMonitor(logging_interval='epoch'),
                                    EarlyStopping(
                                                monitor="Loss/Val",
                                                mode="min",
                                                patience= LR_PATIENCE
                                            ),],
                        gradient_clip_val= 0.5,
                        gradient_clip_algorithm="value",  # https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#gradient-clipping
                        log_every_n_steps=1,
                        num_sanity_val_steps=0,
                        enable_progress_bar=True
                        )
        #trains the model
        trainer.fit(model=model, datamodule=data_module)#, train_dataloaders=dataset
        #saves model`s checkpoint on wandb
        wandb.save(os.path.join(CHECKPOINT_SAVE_PATH, f"{run.name}.ckpt"))
        #saves settings used for that model
        wandb.save(os.path.join(".", "settings.yaml"))

if __name__ == "__main__":
    main()