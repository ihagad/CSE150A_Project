# MDP Results Metrics - sharpyew_kz

Generated from `remote_logs/sharpyew_kz_mdp_training.log` and `results/sharpyew_kz_ep765.pkl`.

## Core Training Metrics

| Metric | Value |
|---|---:|
| Episode range in log | 16–765 |
| Episodes logged | 750 |
| Latest checkpoint | `sharpyew_kz_ep765.pkl` |
| Unique states discovered (latest log) | 7632 |
| Unique states in final checkpoint graph | 7632 |
| Recorded transitions | 374389 |
| Recent 10-episode average reward | -51.9 |
| Best episode reward | 1907.1 |
| Worst episode reward | -12750.0 |
| Positive-reward episodes | 73 |
| Latest PI/VI agreement | 7506/7632 (98.35%) |

## Baseline / Learned Comparison

| Comparison | Average Return |
|---|---:|
| Early high-exploration baseline, first 50 logged episodes | -1442.0 |
| Learned policy, last 50 logged episodes | 2.1 |
| Learned policy, last 10 logged episodes | -51.9 |
| Model-based random policy baseline, 500 rollouts x 100 steps | -148.2 +/- 224.4 |
| Model-based learned policy, 500 rollouts x 100 steps | 423.0 +/- 663.0 |

## Minecraft Progression Evidence

| State/action evidence | Value |
|---|---:|
| States with wood | 1881 |
| States with stone/cobble | 2302 |
| States with wooden tools | 3831 |
| States with stone tools | 2156 |
| States with furnace nearby | 1180 |
| Fuel bucket max / nonzero states | 4 / 2881 |
| Raw iron bucket max / nonzero states | 4 / 325 |
| Iron ingot bucket max / nonzero states | 4 / 126 |
| States with iron pickaxe | 0 |
| Diamond bucket max / nonzero states | 0 / 0 |
| States with diamond pickaxe | 0 |

## Generated Figures

- `results/analysis/sharpyew_kz_rewards.png`
- `results/analysis/sharpyew_kz_growth.png`
- `results/analysis/sharpyew_kz_state_heatmap.png`
