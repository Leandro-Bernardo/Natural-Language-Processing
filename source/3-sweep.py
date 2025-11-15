import torch
import os
import yaml, json
import wandb

from wandb.wandb_run import Run
from models import subj, classn
from models.lightning import DataModule, BaseModel, MultitaskModel

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
    TASK = settings["task"]
    SUBJ_MODEL_VERSION = settings["subj_model"]
    MULTITASK_MODEL_VERSION = settings["multitask_model"]
    # training hyperparams
    MAX_EPOCHS = settings["max_epochs"]
    LR_PATIENCE = settings["learning_rate_patience"]
    TRAIN_SUBJ_WITH_WEIGHTS = settings["train_subj_with_weights"]


# reads sweep configs yaml
with open(os.path.join(os.path.dirname(__file__),'sweep_config.yaml')) as file:
    SWEEP_CONFIGS = yaml.load(file, Loader=yaml.FullLoader)

# models choices
networks_choices = {"subj_classifier": {"subj_classifier_v1": subj.subj_classifier_v1,
                                        "subj_classifier_v2": subj.subj_classifier_v2,
                                        "subj_classifier_v3": subj.subj_classifier_v3},
                      "multitask_model": {"multitask_model_v1": classn.multitask_model_v1,
                                          "multitask_model_v2": classn.multitask_model_v2}}
if TASK == "subj_classifier":
    SUBJ_MODEL_NETWORK = networks_choices[TASK][SUBJ_MODEL_VERSION]
elif TASK == "multitask_classifier":
    SUBJ_MODEL_NETWORK = networks_choices["subj_classifier"][SUBJ_MODEL_VERSION]
    MULTITASK_MODEL_NETWORK = networks_choices["multitask_model"][MULTITASK_MODEL_VERSION]
    SUBJ_MODEL_CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "subj_classifier", f"{SUBJ_MODEL_VERSION}" , f"{settings["subj_chosen_model"]}.ckpt")
else:
    raise Exception(f"Invalid task,\nAvailable: (subj_classifier, multitask_classifier).\nChosen task: {TASK}")

# defines path dir
if TASK == "subj_classifier":
    CHECKPOINT_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", f"{TASK}", f"{SUBJ_MODEL_VERSION}")
elif TASK == "multitask_classifier":
    CHECKPOINT_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", f"{TASK}", f"{MULTITASK_MODEL_VERSION}")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "embeddings")
# creates directories
os.makedirs(CHECKPOINT_SAVE_PATH, exist_ok =True)


dataset_id = {"subj_classifier": 1,
              "multitask_classifier": 2}

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
        # train with weights settings
        subj_majority_weight = configs.get("subj_weight_value", 1.0)
        if TRAIN_SUBJ_WITH_WEIGHTS:
            subj_minority_weight = 1.0 + (1.0 - subj_majority_weight)
            pos_weight_value =  subj_majority_weight / subj_minority_weight
        else:
            pos_weight_value = 1.0
        pos_weight_tensor = torch.tensor(pos_weight_value, dtype=torch.float32)
        loss_choices = {"binary_cross_entropy":torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) if TRAIN_SUBJ_WITH_WEIGHTS else torch.nn.BCEWithLogitsLoss(),
                        "categorical_cross_entropy": torch.nn.CrossEntropyLoss()}
        # load data module
        data_module = DataModule(datasets_root=EMBEDDINGS_PATH, batch_size= configs["batch_size"], dataset_id=dataset_id[TASK], num_workers=6)
        if TASK == "subj_classifier":
            model = BaseModel(model=SUBJ_MODEL_NETWORK, loss_function=loss_choices[configs["subj_loss_function"]], batch_size=configs["batch_size"], learning_rate=configs["lr"], learning_rate_patience=LR_PATIENCE, dataset_id=1)
        elif TASK == "multitask_classifier":
            with open(os.path.join(EMBEDDINGS_PATH, "metadata", "cc_classes.json"), "r") as file:
                cc_classes_metadata = json.load(file)
            num_cc_classes = cc_classes_metadata["total_classes"]
            subj_trained_model = BaseModel.load_from_checkpoint(SUBJ_MODEL_CHECKPOINT,
                                                model=SUBJ_MODEL_NETWORK,
                                                loss_function=torch.nn.BCEWithLogitsLoss(),
                                                strict=False).eval()
            model = MultitaskModel(cc_model = lambda: MULTITASK_MODEL_NETWORK(num_classes=num_cc_classes), subj_trained_model = subj_trained_model,
                                subj_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor), cc_loss = torch.nn.CrossEntropyLoss(),
                                batch_size=16, learning_rate=0.01, learning_rate_patience=LR_PATIENCE, dataset_id=2, num_cc_classes=num_cc_classes)
        else:
            raise Exception(f"Invalid option for task.\nTasks: subj_classifier or multitask_classifier.\nChosen: {TASK}")
        # train the model
        trainer = Trainer(
                        logger= logger,
                        accelerator="cpu",
                        max_epochs=MAX_EPOCHS,
                        callbacks= [checkpoint_callback,
                                    LearningRateMonitor(logging_interval='epoch'),
                                    EarlyStopping(
                                                monitor="acc/Val/Epoch",
                                                mode="max",
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