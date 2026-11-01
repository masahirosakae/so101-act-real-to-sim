# SO-101 ACT Real-to-Sim — Physical AI 2026 応用編1 最終課題

> 実機SO-101で収集した操作データからACTを学習し、MuJoCoで教師関節軌道とAI予測軌道を比較した研究プロジェクトです。
>
> **達成範囲：実機データ収集、ACT学習・推論、MuJoCoでの関節駆動・軌道比較、オフライン定量評価。**
> **未達：ACTによるミニカーの自律的なGoal到達。**

**English summary:** We collected real SO-101 teleoperation demonstrations, trained an ACT imitation-learning policy, and compared teacher joint trajectories with offline ACT predictions in MuJoCo. Autonomous car-to-goal success was **not** demonstrated; the scene props are visual-only and the joint mapping is provisional.

本リポジトリは**研究・検証用のプロトタイプ**です。実機SO-101は教師操作データの収集に使用しましたが、ACTによる実機自律制御は行っていません。実機のCOMポートやサーボに接続せず、シミュレーション・オフライン評価を実行する構成です。

- [最終レポート（PDF）](report/SO101_Real_to_Sim_Final_Report.pdf)
- 提出動画：2分17秒（YouTubeに別途提出。リポジトリには動画の比較サンプルのみ同梱）

![Episode 2 フレーム70の実機・MuJoCo姿勢比較](figures/real_sim_comparison.png)

*同一エピソードのframe 70を比較。カメラ画角と配置は目視で近似しており、実機座標系の厳密なキャリブレーションを完了したものではありません。*

## 1. 研究目的

対象タスクは、SO-101ロボットアームを操作して、**青いミニカーを赤いGoal枠内へ移動すること**です。実機Leader/Followerによる教師操作を記録し、Transformerベースの模倣学習モデルACT（Action Chunking with Transformers）を学習しました。

実機のACT制御では安全な起動方法を確立できなかったため、評価対象をMuJoCoの仮想SO-101に移行しました。今回の目的は、教師関節軌道とAI予測関節軌道の差、およびReal-to-Simの技術的課題を明確にすることです。

## 2. 実装した内容

- **実機データの収集**：10エピソード × 450フレーム = **4,500フレーム**（15 FPS）。上面カメラ640×480、6関節の状態（state）と教師アクション（action）をLeRobot Dataset v3.0形式で記録しました。
- **ACTモデルの学習**：No-VAE構成を新規学習し、5,000ステップのチェックポイントから10,000ステップまで再開。6,000・8,000・10,000ステップを比較し、**6,000ステップを代表モデル**としました。
- **MuJoCo環境**：SO-101 MJCFを読み込み、青いミニカーと赤いGoal枠を画像上の配置に近づけて可視化しました。
- **軌道の比較**：教師指令とACTの**オフライン推論**によるアクションを、それぞれMuJoCo上で再生し、関節動作を比較しました。
- **固定画像でのACT推論試験**：録画済みの実写画像1枚を固定し、毎ステップ更新される仮想関節状態をACTへ入力する試験を実施しました。30秒試験では開始直後に小さく動いた後、ほぼ静止しました。**シミュレータのカメラ画像による視覚閉ループ制御ではありません。**

**公開範囲：** 実機の元データセットと学習済み重みは本リポジトリに含めていません。

## 3. ACT予測精度の評価

Episode 2の450フレームを用い、各時刻の実機画像と関節状態をACTに入力し、**チャンク先頭の予測アクション**と同時刻の教師アクションの平均絶対誤差（MAE）を比較しました。静止ベースラインは、現在のstateをそのまま次の目標アクションとする単純な予測です。

| 関節 | ACT MAE | 静止ベースライン MAE |
|---|---:|---:|
| shoulder_pan | 4.887° | 1.864° |
| shoulder_lift | 7.502° | 2.410° |
| elbow_flex | 5.497° | 2.921° |
| wrist_flex | 1.977° | 1.219° |
| wrist_roll | 0.773° | 0.424° |
| **5軸の単純平均** | **4.126°** | **1.768°** |

**ACTの誤差は静止ベースラインの約2.33倍であり、同時刻の予測精度はベースラインを下回りました。** グリッパー観測値はほぼ一定で、アクションとの単位・スケールも異なるため、上記の角度平均には含めていません。

### 時間差（ラグ）の探索的評価

420フレームの共通区間について、ACT予測[t]と教師アクション[t+lag]を比較しました。

- lag=0：ACT MAE **4.262°**
- lag=+9フレーム（約0.60秒）：ACT MAE **2.923°**
- 同じlag=+9の教師時刻に対する静止ベースライン：**5.040°**
- 教師指令[t]と実機関節状態[t+lag]の最適ラグ：全体で約**+3フレーム（0.20秒）**

ラグは評価対象の系列を使って事後的に探索した値です。**時間同期の不具合や将来予測の汎化性能が証明されたわけではありません。** 詳細は[`results/temporal_audit.md`](results/temporal_audit.md)を参照してください。

評価対象のEpisode 2は、後述の単一タスク・10エピソードという分割条件なら**学習側**です。未知エピソードの成績やタスク成功率としては解釈できません。

## 4. 実証できていない事項と制約

1. **MuJoCo上でのミニカーの自律的なGoal到達は未達成**です。ミニカーとGoalは表示専用の静的オブジェクトで、接触・押し込み物理や成功判定は実装していません。
2. 2画面比較動画のACT側は、**実機の観測から事前計算したオフライン予測を再生したもの**であり、仮想環境を認識する自律ポリシーのロールアウトではありません。
3. 関節の単位変換（度→ラジアン）は実装しましたが、**符号+1、オフセット0は暫定設定**です。+15°の補正候補も試しましたが、複数フレームで一貫した改善が得られず採用していません。
4. 学習用・評価用エピソードの実際の割当履歴、未学習データでの汎化性能、No-VAE側の完全なLoss曲線は未確認です。
5. 教師操作の実機映像ではミニカーがGoal内へ移動しましたが、これは**人間のテレオペレーションによる結果**です。AIによる自律成功を示すものではありません。

## 5. 学習条件とモデルの由来

チェックポイントの保存設定、`training_step.json`、学習実行時のPowerShell履歴から確認した内容です。

| 項目 | 設定・確認結果 |
|---|---|
| フレームワーク | LeRobot v0.6.1、ACT **No-VAE** |
| 画像バックボーン | ResNet18 |
| Transformer | `dim_model=128`、Encoder 2層、Decoder 1層 |
| Action Chunk | `chunk_size=20`、保存時`n_action_steps=20` |
| 評価時設定 | `n_action_steps=1`（各時刻におけるチャンク先頭を比較） |
| Optimizer | AdamW、学習率`1e-5`、weight_decay`1e-4` |
| AdamW詳細 | betas `(0.9, 0.999)`、eps `1e-8`、gradient clipping `10.0` |
| バッチ・乱数 | `batch_size=1`、`seed=1000` |
| ログ | `log_freq=100`、W&B無効 |
| Scheduler | 保存設定では未指定 / null |
| 画像拡張 | `dataset.image_transforms.enable=false`、`use_imagenet_stats=true` |
| 分割設定 | `eval_split=0.2` |
| 最初のNo-VAE学習 | `act_fullgoal_novae_fresh_5000_20261101_192814`、5,000ステップ時点で`resume=false`、`checkpoint_path`・`pretrained_path`は指定なし |
| No-VAE再開学習 | 上記005000の設定から`--resume=true --steps=10000`で再開。006000〜010000チェックポイントを保存 |
| 採用モデル | No-VAE、6,000ステップ |
| 別の実験 | VAEありの1,000〜5,000ステップ。VAEありのLossをNo-VAEの学習曲線には混在させません |
| 未確認 | No-VAEのLoss履歴、厳密な学習時間・使用GPU、ResNet18の初期重みの由来 |

### 評価エピソードの分割について

LeRobot v0.6.1では`eval_split=0.2`の場合、**タスクごとに末尾の20%（端数切り上げ）のエピソード**を評価側に分ける仕様です。全10エピソードが同一タスクであれば、Episode 0〜7が学習側、8〜9が評価側です。ただし、各エピソードのtask割当と実際の学習ログをここでは照合していません。

- [LeRobot v0.6.1の分割実装](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/factory.py)
- [学習設定の追加記録](TRAINING_METADATA_REQUIRED.md)

### 保存されているNo-VAE再開学習コマンド

```powershell
$CheckpointConfig = 'D:\SO101\outputs\act_fullgoal_novae_fresh_5000_20261101_192814\checkpoints\005000\pretrained_model\train_config.json'
& 'C:\SO101\.venv-act\Scripts\lerobot-train.exe' "--config_path=$CheckpointConfig" '--resume=true' '--steps=10000' '--output_dir=D:\SO101\outputs\act_fullgoal_novae_resume_5000_to_10000' '--job_name=act_fullgoal_novae_resume_10000'
```

これは履歴に残っていた**再開学習**コマンドです。最初のNo-VAE新規学習コマンドそのものではありません。

## 6. リポジトリ構成

```text
configs/       関節変換の暫定マッピング
src/           MuJoCo、ACT評価、軌道比較、時間差解析
results/       予測精度・チェックポイント比較・ラグ解析結果
figures/       実機とMuJoCoの比較画像
media/         軌道比較のサンプル動画
report/        最終レポートPDF
tests/         静的な安全性チェック
```

## 7. 再現手順（Windows PowerShell）

**注意：** ACTとMuJoCoは**別のPython仮想環境**です。以下のパスは実験時のWindows環境に基づいており、利用するPCに合わせて変更する必要があります。サーボやCOMポートを操作するコマンドは含みません。

```powershell
$Repo = 'D:\SO101\so101-act-real-to-sim'
$SimPy = 'C:\SO101\.venv-mujoco\Scripts\python.exe'
$ActPy = 'C:\SO101\.venv-act\Scripts\python.exe'
$Scene = 'C:\SO101\sim_models\SO-ARM100\Simulation\SO101\scene.xml'
$Data = 'D:\SO101\datasets\so101_car_fullgoal_10ep_20261101_100029'
$Ckpt = 'D:\SO101\outputs\act_fullgoal_novae_resume_5000_to_10000\checkpoints\006000\pretrained_model'
$Map = Join-Path $Repo 'configs\mapping_candidate_UNVERIFIED.json'

# 関節名・可動範囲の確認（シミュレーションのみ）
& $SimPy (Join-Path $Repo 'src\so101_sim.py') --mode inspect --scene $Scene

# 表示専用ミニカー・GoalのMJCFを作成
& $SimPy (Join-Path $Repo 'src\create_so101_car_goal_scene.py') --scene $Scene --car-x 0.20 --car-y -0.08 --goal-x 0.37 --goal-y -0.08

# 実機画像に近づけた配置と仮想カメラを追加
$Visual = Join-Path (Split-Path $Scene) 'scene_car_goal_visual.xml'
& $SimPy (Join-Path $Repo 'src\so101_align_car_goal_to_real.py') --scene $Visual

# ACTのオフライン予測（実機データセット・学習済み重みが別途必要）
& $ActPy (Join-Path $Repo 'src\so101_act_offline_eval.py') --dataset $Data --checkpoint $Ckpt --episode 2 --stride 1 --output 'D:\SO101\offline_eval' --device cpu

# 固定実写画像+仮想関節状態のACT推論試験。視覚閉ループではない
& $SimPy (Join-Path $Repo 'src\so101_sim.py') --mode act --scene $Scene --act-python $ActPy --dataset $Data --checkpoint $Ckpt --mapping $Map --dataset-frame 1218 --frames 45 --max-step-rad 0.02 --output 'D:\SO101\fixed_image_rollout'

# 教師軌道とACTオフライン予測をMuJoCoで再生
$Pred = 'D:\SO101\offline_eval\predictions.csv'
$VisualScene = 'C:\SO101\sim_models\SO-ARM100\Simulation\SO101\scene_car_goal_real_layout.xml'
& $SimPy (Join-Path $Repo 'src\so101_teacher_vs_act_replay.py') --csv $Pred --scene $VisualScene --mapping $Map --start-frame 70 --frames 300 --fps 15 --max-step-rad 0.10 --camera car_goal_top --video 'D:\SO101\teacher_vs_act.mp4'
```

## 8. 再現性・公開上の注意

- `results/ep2_predictions.csv` と `results/ep2_metrics.json` は**本リポジトリに含まれていません**。元データ・重み・ローカル解析結果がない状態で完全再現可能とは主張しません。詳細は[`results/IMPORT_LOCAL_RESULTS.md`](results/IMPORT_LOCAL_RESULTS.md)を参照してください。
- 実機の録画データ、人物情報、秘密鍵、アクセストークン、COMログ、モデルの重み、キャッシュ等は公開しないでください。
- ロボットのMJCF・meshとLeRobotのモデルは、各配布元のライセンスを確認したうえで取得してください。外部モデルや学習済み重みは同梱していません。
- このリポジトリのオリジナルコードについて、現時点で利用許諾ライセンス（LICENSE）は選定していません。

### 参考・謝辞

- [LeRobot（Hugging Face）](https://github.com/huggingface/lerobot)
- [SO-101 MJCF（TheRobotStudio / SO-ARM100）](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101)
- [MuJoCoドキュメント](https://mujoco.readthedocs.io/)
