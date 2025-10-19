# Str-DiffSep: Streamable Diffusion Model for Speech Separation

This is the repository for Streamable Diffusion Model for Speech Separation.  

## Environment
Use `dit_new.yml` to create conda environment.  
```
conda env create -f dit_new.yml
conda activate dit_new
```

## Configuration
You can set the configuration of the model and the dataset by modifiying ``.ymal`` files in config folder. Configuration is done using the hydra hierarchical configuration package. The hierarchy is as follows.  

```
config/
|-- config.yaml  # main config file
|-- datamodule  # config of dataset and dataloaders
|   |-- default.yaml  
|-- model
|   |-- default.yaml  # original NCSN++ model
|   |-- skim.yaml  # Str-DiffSep, SkiM-based score function backbone
`-- trainer
    `-- default.yaml  # config of pytorch-lightning trainer
    `-- allgpus.yaml  # config using GPU
```

## Dataset
The wsj0_mix dataset is expected in data/wsj0_mix

```
data/wsj0_mix/
|-- 2speakers
|   |-- wav16k
|   |   |-- max
|   |   |   |-- cv
|   |   |   |-- tr
|   |   |   `-- tt
|   |   `-- min
|   |       |-- cv
|   |       |-- tr
|   |       `-- tt
|   `-- wav8k
|       |-- max
|       |   |-- cv
|       |   |-- tr
|       |   `-- tt
|       `-- min
|           |-- cv
|           |-- tr
|           `-- tt
`-- 3speakers
    |-- wav16k
    |   `-- max
    |       |-- cv
    |       |-- tr
    |       `-- tt
    `-- wav8k
        `-- max
            |-- cv
            |-- tr
            `-- tt
```


The Libri2Mix dataset is expected in data/LibriMix/Libri2Mix.
```
LibriMix/
|-- Libri2Mix
|   |-- wav8k
|   |   |-- max
|   |   |   |-- dev
|   |   |   |-- metadata
|   |   |   |-- test
|   |   |   |   |-- mix_both
|   |   |   |   |-- mix_clean
|   |   |   |   |-- mix_single
|   |   |   |   |-- noise
|   |   |   |   |-- s1
|   |   |   |   `-- s2
|   |   |   |-- train-100
|   |   |   `-- train-360
|   |   `-- min
|   |       |-- dev
|   |       |-- metadata
|   |       |-- test
|   |       |-- train-100
|   |       `-- train-360
|   `-- wav16k
|       |-- max
|       |-- min
```

## Training and Evaluation
Use `python train.py` to start training the model.  

Use `python envaluate.py` to select sampler, test the model and get inference results.  
