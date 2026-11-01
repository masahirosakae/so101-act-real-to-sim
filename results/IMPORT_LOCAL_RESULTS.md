# ローカル結果の追加（未収録）

完全なオフライン監査再現にはWindows PCの以下の2ファイルが必要です。**このZIPには含まれていません。**

```powershell
$Repo = 'D:\SO101\so101-act-real-to-sim'
$Src = 'D:\SO101\real_to_sim_results\offline_ep2_all450'
Copy-Item "$Src\predictions.csv" "$Repo\results\ep2_predictions.csv"
Copy-Item "$Src\metrics.json" "$Repo\results\ep2_metrics.json"
```

注意：`metrics.json` 内に `checkpoint`、`dataset` のローカル絶対パスが含まれるため、公開前にパス・個人情報を確認してください。CSV自体も実機関節軌道を含むのでライセンス/情報公開の判断が必要です。
