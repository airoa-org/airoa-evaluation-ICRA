# 実機運用メモ

このメモは，今の構成で魔改造HSRを動かす時に，推論PCと制御PCで何をやるかをざっくりまとめた物です．  

## 1. 推論PCですること

リポジトリへ移動して，checkpoint と port をセットして server を起動します．

```bash
cd ~/usr/icra_vla_ws/airoa-evaluation-ICRA-ROS1

export POLICY_CHECKPOINT_PATH=/abs/path/to/checkpoint_dir
# export POLICY_CHECKPOINT_PATH=/media/hma/TeruSSD/checkpoints/0412_pi05_airoa_hsr_fullfinetuning_statediff_horizon8_relocate/0412_pi05_airoa_hsr_fullfinetuning_statediff_horizon8_relocate/5000
export POLICY_CONFIG_NAME=config_name
# export POLICY_CONFIG_NAME=0412_pi05_airoa_hsr_fullfinetuning_statediff_horizon8_relocate
export POLICY_SERVER_PORT=8000
```
*[server/entry_point.sh](../server/entrypoint.sh)*のPOLICY_CONFIG_NAMEを修正する
※ ROS1版はここを修正しないとサーバー側のPOLICY_CONFIG_NAMEが書き換わらない
```shell
POLICY_CONFIG_NAME="${POLICY_CONFIG_NAME:-pi0_hsr_airoa-moma}"
↓
POLICY_CONFIG_NAME="${POLICY_CONFIG_NAME:-0412_pi05_airoa_hsr_fullfinetuning_statediff_horizon8_relocate}"
```

`POLICY_CONFIG_NAME` は checkpoint を学習したときの config 名と一致させてください．
server はこの config から `action_mode` を metadata として配信し，client はそれを使って arm/head の action 解釈を切り替えます．
ここがずれると `absolute` や `state_diff` の deploy が崩れます．


## 2. 制御PCですること

```bash
cd ~/usr/icra_vla_ws/airoa-evaluation-ICRA-ROS1

# for B022
export HSR_ID=B022
export HSR_IP=192.168.0.2
export ROS_MASTER_URI=http://192.168.0.2:11311
export ROS_IP=192.168.0.10
export TEST_MODE=false
export POLICY_SERVER_HOST=172.30.21.164 # HMA WiFi 5
export POLICY_SERVER_PORT=8000
"""

```
そのあと client container を起動します．

```bash
./RUN-DOCKER-CONTAINER.sh down
./RUN-DOCKER-CONTAINER.sh up
```

必要ならログ確認です．

```bash
./RUN-DOCKER-CONTAINER.sh logs policy_server
```

そのあと シェルに入ります．

```bash
./RUN-DOCKER-CONTAINER.sh shell
```

## 3. 実際に client を起動する

制御PCの shell の中でこれを実行します．

```bash
roslaunch hsr_policy_client hsr_policy_client.launch \
  test_mode:=false \
  instruction:='Pick up an object'

roslaunch hsr_policy_client hsr_policy_client.launch \
  test_mode:=false \
  action_mode:=state_diff_arm_head_relative_gripper_base \
  instruction:='From a rectangle by relocating the mug that is not at a rectangle corner 3' 

  
roslaunch hsr_policy_client hsr_policy_client.launch \
  test_mode:=false \
  action_mode:=state_diff_arm_head_relative_gripper_base \
  instruction:='pick up the coffee bottle on the right' 
```

今の launch default は保守寄りです．
`adopted_action_chunks:=1`, `upsample:=false`, `action_smoothing:=none` を既定にしてあり，
特に `absolute` / `state_diff` の最初の切り分け向けにしています．
server metadata が取れない古い構成を使う場合だけ，必要に応じて
`action_mode:=relative` / `action_mode:=absolute_arm_head_relative_gripper_base` /
`action_mode:=state_diff_arm_head_relative_gripper_base` を明示してください．
