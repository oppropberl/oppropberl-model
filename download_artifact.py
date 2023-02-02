import wandb

run = wandb.init(project="annotation-prediction", tags=[
                     "initial-setup"], notes="complete PyG model v0.1", settings=wandb.Settings(start_method='thread'))

artifact = run.use_artifact(
        'anonymous/annotation-prediction/fuzzed-df:latest', type='dataset')

artifact_dir = artifact.download()

print(artifact_dir)