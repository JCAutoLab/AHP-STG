# PatchSTG Optimization Log

## SD future-time decoder

- Config: `config/SD_future.conf`
- Baseline source: `sd_adaptive_stage3_model`
- Best validation MAE: 15.5562 at epoch 25
- Test MAE/RMSE/MAPE: 16.9721 / 28.9588 / 0.1092
- Conclusion: future-time aware decoder is a clear improvement over the previous SD stage3 test MAE 17.5953.

## SD temporal token ablation

- `config/SD_temporal_tpn2.conf`: `tps=6`, `tpn=2`, temporal mixer on.
  - Best validation MAE: 15.4966 at epoch 19
  - Test MAE/RMSE/MAPE: 17.4898 / 30.4268 / 0.1154
- `config/SD_temporal_tpn4.conf`: `tps=3`, `tpn=4`, temporal mixer on.
  - Best validation MAE: 15.5637 at epoch 18
  - Test MAE/RMSE/MAPE: 18.1536 / 32.2460 / 0.1232
- Conclusion: directly increasing temporal tokens did not improve SD test accuracy. The likely reason is that changing `tps/tpn` skips pretrained `input_st_fc` and `regression_conv` weights, weakening transfer.

## SD graph bias and soft merge ablation

- `config/SD_graph.conf`: graph bias scale 0.05.
  - Best validation MAE: 15.3526 at epoch 15
  - Test MAE/RMSE/MAPE: 16.7993 / 28.6363 / 0.1080
- `config/SD_graph_s01.conf`: graph bias scale 0.10.
  - Best validation MAE: 15.3330 at epoch 15
  - Test MAE/RMSE/MAPE: 16.8202 / 28.6940 / 0.1081
- `config/SD_graph_s002.conf`: graph bias scale 0.02.
  - Best validation MAE: 15.3376 at epoch 15
  - Test MAE/RMSE/MAPE: 16.7815 / 28.6131 / 0.1079
- `config/SD_softmerge.conf`: soft merge temperature 1.5, target 0.35, reset refiner residual scale.
  - Best validation MAE: 15.3493 at epoch 15
  - Test MAE/RMSE/MAPE: 16.7941 / 28.6283 / 0.1080
- `config/SD_graph_softmerge.conf`: graph bias scale 0.05 plus soft merge.
  - Best validation MAE: 15.3346 at epoch 15
  - Test MAE/RMSE/MAPE: 16.7946 / 28.6344 / 0.1080
- `config/SD_graph_lowlr.conf`: graph bias scale 0.05, lr 2e-5.
  - Best validation MAE: 15.5478 at epoch 4
  - Test MAE/RMSE/MAPE: 16.8694 / 28.7638 / 0.1083
- `config/SD_softmerge_lowlr.conf`: soft merge, lr 2e-5.
  - Best validation MAE: 15.5638 at epoch 4
  - Test MAE/RMSE/MAPE: 16.8941 / 28.7981 / 0.1085
- Conclusion: graph attention bias is the best next improvement on SD. The strongest run is `SD_graph_s002`, improving test MAE from 16.9721 to 16.7815. Soft merge also improves SD, but the graph-only variant is slightly better.

## Cross-dataset graph bias and soft merge extension

- CA baseline `config/CA_future.conf`
  - Best validation MAE: 15.3618 at epoch 28
  - Test MAE/RMSE/MAPE: 17.2775 / 29.6577 / 0.1222
- `config/CA_graph.conf`
  - Best validation MAE: 15.3196 at epoch 16
  - Test MAE/RMSE/MAPE: 17.2660 / 29.6655 / 0.1220
- `config/CA_softmerge.conf`
  - Best validation MAE: 15.3197 at epoch 16
  - Test MAE/RMSE/MAPE: 17.2652 / 29.6628 / 0.1219
- CA conclusion: soft merge is the best of this round, with a small but real test MAE improvement from 17.2775 to 17.2652.

- GBA baseline `config/GBA_future.conf`
  - Best validation MAE: 17.4345 at epoch 29
  - Test MAE/RMSE/MAPE: 19.7092 / 33.2876 / 0.1512
- `config/GBA_graph.conf`
  - Best validation MAE: 17.3053 at epoch 18
  - Test MAE/RMSE/MAPE: 19.7233 / 33.4597 / 0.1503
- `config/GBA_graph_s0005.conf`
  - Best validation MAE: 17.3017 at epoch 18
  - Test MAE/RMSE/MAPE: 19.7194 / 33.4544 / 0.1502
- `config/GBA_softmerge.conf`
  - Best validation MAE: 17.3221 at epoch 19
  - Test MAE/RMSE/MAPE: 19.8154 / 33.5868 / 0.1507
- GBA conclusion: the new modules improve validation MAE and MAPE but do not beat the future baseline on test MAE. Keep `GBA_future` as the recommended GBA checkpoint for now.

- GLA baseline `config/GLA_future.conf`
  - Best validation MAE: 17.0649 at epoch 30
  - Test MAE/RMSE/MAPE: 19.3136 / 32.4799 / 0.1173
- `config/GLA_graph.conf`
  - Best validation MAE: 16.8159 at epoch 19
  - Test MAE/RMSE/MAPE: 19.2177 / 32.5349 / 0.1153
- `config/GLA_softmerge.conf`
  - Best validation MAE: 16.8367 at epoch 19
  - Test MAE/RMSE/MAPE: 19.1991 / 32.3929 / 0.1157
- GLA conclusion: soft merge is the best by test MAE, improving from 19.3136 to 19.1991. Graph bias has the best validation MAE and MAPE, but slightly worse test MAE than soft merge.

## Final recommended checkpoints

- SD: `./cpt/sd_graph_s002_model` from `config/SD_graph_s002.conf`.
- CA: `./cpt/ca_softmerge_model` from `config/CA_softmerge.conf`.
- GBA: `./cpt/gba_future_model` from `config/GBA_future.conf`.
- GLA: `./cpt/gla_softmerge_model` from `config/GLA_softmerge.conf`.
