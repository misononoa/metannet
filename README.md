# metannet

マイクで話した声を Whisper で認識し、VOICEVOX で別の声に読み上げ直すツール(配信などで自分の声を出したくないとき用)。

## 使い方

### podman-compose(VOICEVOX も同時に起動)

```bash
podman-compose up
```

初回は VOICEVOX イメージと Whisper モデルを取得します。`Ctrl-C` で停止。

認識結果の表示のみ(読み上げなし):

```bash
podman-compose up -d voicevox
podman-compose run --rm app uv run main.py --transcribe-only
```

### 直接実行

別途 VOICEVOX ENGINE を `:50021` で起動した状態で:

```bash
uv run main.py                     # マイク → 認識 → 読み上げ
uv run main.py --transcribe-only   # 認識結果の表示のみ
uv run main.py --list-devices      # 音声デバイス一覧
```

主なオプション: `--model`(Whisper モデル名), `--speaker`(VOICEVOX 話者ID), `--input-device` / `--output-device`, `--voicevox-url`。
