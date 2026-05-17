$ErrorActionPreference = "Stop"

$baseConfig = "configs/baseline_resnet18_pretrained.yaml"

$experiments = @(
  "experiment_name=hparam_lr_1e3 output_dir=outputs/hparam_lr_1e3 train.lr=0.001",
  "experiment_name=hparam_lr_3e4 output_dir=outputs/hparam_lr_3e4 train.lr=0.0003",
  "experiment_name=hparam_bs16 output_dir=outputs/hparam_bs16 train.batch_size=16",
  "experiment_name=hparam_bs32 output_dir=outputs/hparam_bs32 train.batch_size=32",
  "experiment_name=hparam_ep10 output_dir=outputs/hparam_ep10 train.epochs=10",
  "experiment_name=hparam_ep20 output_dir=outputs/hparam_ep20 train.epochs=20"
)

python src/sweep.py --config $baseConfig --overrides $experiments
