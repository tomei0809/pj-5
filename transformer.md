# transformer.md ― Transformer 担当のメモ

`systems/transformer.py` に、次の3つの変更を入れるための計画。

- ① 改善
- ② ファインチューニングの形（encode / head）に合わせる
- ③ モダリティに分ける

**進める順番は ② → ③ → ①。**

1. ②は形を変えるだけで、出力は変わらない
2. ③で、モダリティごとに設定を変えられる形にする
3. ①は、③で分けたファイルごとに1つずつ試す

---

## 1. 懸念点

| 懸念                                                 | 何が困るか                                                                    | 対応                                                              |
| ---------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `build_model(in_dim)` にモダリティが届かない         | 3つとも768次元なので、1つのファイルの中ではモダリティごとに設定を変えられない | ③でファイルを分ける（`transformer_text.py` など）                 |
| ファイルとモダリティの取り違え                       | `transformer_text` を `--modality visual` で学習してしまう                    | `train_one_epoch` の最初で `args.modality` を確かめ、違えば止める |
| 結果のフォルダ名が重なる                             | `runs_mosu/transformer_text_text_seed0/` になる                               | 見た目だけの問題。そのままにする                                  |
| 設定を変えると、前の `best.pt` が読めない            | 層を足すと重みの名前が変わる                                                  | 渡すときは「システム名・`best.pt`・コミットID」をセットにする     |
| `ja_holdout.txt` がまだない                          | 学習し直すデータが作れない                                                    | ファインチューニング担当と、作る人と置き場所を確かめる            |
| 共通の `train.py` を直してモダリティを渡す方法もある | 全員のコードに関わる                                                          | チームの相談が要るので、今回はしない                              |

---

## 2. ② ファインチューニングの形に合わせる

`forward` の中身を `encode` に移す。

**変更前**（今の [systems/transformer.py:43-47](systems/transformer.py#L43-L47)）

```python
    def forward(self, X, M):
        # X:（バッチ, 秒数, 次元） M:（バッチ, 秒数） 有効な秒が True
        h = self.inp(X) * math.sqrt(D_MODEL) + positional_encoding(X.shape[1], D_MODEL, X.device)
        h = self.encoder(self.drop(h), src_key_padding_mask=~M)  # 詰め物（False）は注意の対象から外す
        return self.head(self.norm(h)).squeeze(-1)  # （バッチ, 秒数）のスコア
```

**変更後**

```python
    def encode(self, X, M):
        # X:（バッチ, 秒数, 768） M:（バッチ, 秒数） 本物の秒が True
        h = self.inp(X) * math.sqrt(D_MODEL) + positional_encoding(X.shape[1], D_MODEL, X.device)
        h = self.encoder(self.drop(h), src_key_padding_mask=~M)  # 詰め物（False）は注意の対象から外す
        return self.norm(h)  # （バッチ, 秒数, 128）

    def forward(self, X, M):
        return self.head(self.encode(X, M)).squeeze(-1)  # head の値をそのまま返す（sigmoid なし）
```

- `head` は今の `nn.Linear(D_MODEL, 1)`（= `Linear(128, 1)`）のままでよい
- 層の名前（`inp`・`encoder`・`norm`・`head`）は変わらない。そのため、今の `best.pt` もそのまま読めて、出力も変わらない
- 今の Transformer にはもともと sigmoid がないので、**②だけなら学習し直さなくてよい**（GRU は sigmoid を外すので学習し直しが必要）

---

## 3. ③ モダリティに分ける

### ファイルの分け方

| ファイル                        | 中身                                                                   |
| ------------------------------- | ---------------------------------------------------------------------- |
| `systems/transformer.py`        | 本体。設定を辞書で受け取る。単体で使うと今の設定になる（改善前の基準） |
| `systems/transformer_visual.py` | 映像用の設定だけ（数行）                                               |
| `systems/transformer_audio.py`  | 音声用の設定だけ                                                       |
| `systems/transformer_text.py`   | 字幕用の設定だけ                                                       |

- `D_MODEL = 128` は全モダリティで固定する（encode の出力は128次元という決まりのため）
- 学習するときは、`SYSTEM=` と `MODALITY=` をそろえる（例：`SYSTEM=transformer_text,MODALITY=text`）

### transformer.py の変更（②の後からの差分）

```diff
+BASE = dict(n_layers=2, dropout=0.1, in_norm=False, conv=False)  # 今の設定

 class HighlightTransformer(nn.Module):
-    def __init__(self, in_dim):
+    def __init__(self, in_dim, cfg=None):
         super().__init__()
+        c = {**BASE, **(cfg or {})}
         self.inp = nn.Linear(in_dim, D_MODEL)
-        self.drop = nn.Dropout(DROPOUT)
-        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_FF, DROPOUT,
+        self.in_norm = nn.LayerNorm(D_MODEL) if c["in_norm"] else None              # ①で使う
+        self.conv = nn.Conv1d(D_MODEL, D_MODEL, 3, padding=1) if c["conv"] else None  # ①で使う
+        self.drop = nn.Dropout(c["dropout"])
+        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_FF, c["dropout"],
                                            batch_first=True, norm_first=True)
-        self.encoder = nn.TransformerEncoder(layer, N_LAYERS, enable_nested_tensor=False)
+        self.encoder = nn.TransformerEncoder(layer, c["n_layers"], enable_nested_tensor=False)

     def encode(self, X, M):
-        h = self.inp(X) * math.sqrt(D_MODEL) + positional_encoding(X.shape[1], D_MODEL, X.device)
+        h = self.inp(X)
+        h = self.in_norm(h) if self.in_norm is not None else h * math.sqrt(D_MODEL)
+        h = h + positional_encoding(X.shape[1], D_MODEL, X.device)
+        if self.conv is not None:
+            h = h * M.unsqueeze(-1)                                   # 詰め物の秒を0にする
+            h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)      # 前後1秒を混ぜて足す
         h = self.encoder(self.drop(h), src_key_padding_mask=~M)
         return self.norm(h)

-def build_model(in_dim):
-    return HighlightTransformer(in_dim)
-
-def train_one_epoch(model, data, opt, args, dev, rng):
+def make(cfg=None, modality=None):
+    """設定とモダリティから build_model と train_one_epoch を作る"""
+    def build_model(in_dim):
+        return HighlightTransformer(in_dim, cfg)
+
+    def train_one_epoch(model, data, opt, args, dev, rng):
+        if modality and args.modality != modality:
+            raise ValueError(f"このシステムは {modality} 用（--modality {args.modality} が指定された）")
         ...（中身は今と同じ）
+    return build_model, train_one_epoch
+
+build_model, train_one_epoch = make()  # SYSTEM=transformer のときは今の設定
```

- `BASE` のままなら、作られる層は②と同じ。そのため `SYSTEM=transformer` の結果と `best.pt` は変わらない
- `N_LAYERS` と `DROPOUT` の定数は `BASE` に移す。`N_HEAD`・`D_FF`・`CLIP` は定数のまま

### モダリティごとのファイル（例：transformer_text.py）

```python
from systems.transformer import make

build_model, train_one_epoch = make({"conv": True}, modality="text")
```

- ①で試す設定は、ここの辞書だけを書き替える
- ファインチューニング側は `systems.transformer_text` を読み込めば、`build_model(768)` で同じ形のモデルを作り、`best.pt` を読める

---

## 4. ① 改善の案

今の ρ は、EI.md 8章の結果（cod / game）。

| モダリティ | 今の ρ        | 参考：GRU     | まず試すこと                                                  | 書き替える設定                                     |
| ---------- | ------------- | ------------- | ------------------------------------------------------------- | -------------------------------------------------- |
| 映像       | 0.139 / 0.176 | 0.406 / 0.383 | 入力の √128 倍をやめて LayerNorm にする                       | `in_norm=True`                                     |
| 音声       | 0.121 / 0.217 | 0.392 / 0.360 | 上の LayerNorm ＋ 過学習対策。`best_epoch` が早すぎないか見る | `in_norm=True`、次に `dropout=0.3` か `n_layers=1` |
| 字幕       | 0.369 / 0.307 | 0.424 / 0.361 | 前後の秒を混ぜる畳み込み（字幕のない秒が多いため）            | `conv=True`、次に `in_norm=True`                   |

- **全モダリティ共通で一番怪しいのは、入力を約11倍にしていること**（EI.md 8.5.4 の 1）。位置の情報が埋もれ、学習が暴れやすい
- 原因の切り分け：`summary.json` の `log` で、loss が下がって val_error が上がるなら過学習、loss も下がらないなら学習が不安定
- **使わないもの**
  - sigmoid：head の後ろに関数を入れない決まりがあるため
  - ウォームアップ：lr は全員共通なので、相談してから

**試し方**

1. 1回に1つだけ変える
2. cod・シード0 で試す
3. 良ければ game と 3シードで確かめる
4. `highlight_eval.py compare` で、`transformer`（改善前）と並べて比べる

---

## 5. 確かめ方

| いつ     | 確かめること                                                                    |
| -------- | ------------------------------------------------------------------------------- |
| ②の後    | 同じ `best.pt` を読んで、変更前と予測が一致する                                 |
| ③の後    | `SYSTEM=transformer` の `gpu_test` でパラメータ数が今と同じ（363777）           |
| ③の後    | `SYSTEM=transformer_text,MODALITY=visual` で、エラーで止まる                    |
| ①の後    | `compare` の ρ が改善前より上がっている。シード間のばらつきも見る               |
| 渡すとき | システム名・`best.pt`・コミットIDをセットにして、ファインチューニング担当に渡す |

---

## 進捗のまとめ

### 行ったこと

transformer.py に対して、次の①〜③の変更を行いました。まだ元のリポジトリには反映させていません。

**① 改善**

- 入力を約11倍（√128倍）にする処理をやめ、LayerNorm で大きさをそろえる形に変更。位置の情報が埋もれて学習が不安定になっている可能性への対策
- 音声は、過学習対策として dropout の引き上げ（0.1 → 0.3）と層の数の削減（2 → 1）を試す設定
- 字幕は、前後1秒を混ぜる畳み込みを追加。字幕のない秒が多いため、近くの秒の変化を拾うための工夫
- sigmoid は、head の後ろに関数を入れない決まりに合わせて不使用

**② ファインチューニングの方式（encode / head）への対応**

- forward の中身を encode(X, M) に移動。出力は（バッチ, 秒数, 128）
- head は Linear(128, 1) のみで、後ろに sigmoid などの関数はなし
- forward は head(encode(X, M)) の値をそのまま返す形
- 層の名前は変えていないため、これまでの best.pt もそのまま読み込み可能。もともと sigmoid を使っていないので、出力も変更前と同じ

**③ モダリティごとの分割**

- transformer.py 本体を、設定（層の数・dropout・LayerNorm・畳み込みの有無）を受け取ってモデルを作る形に変更
- モダリティごとの設定は、transformer_visual.py・transformer_audio.py・transformer_text.py に分けて記述（各数行）
- 学習時は `SYSTEM=transformer_text,MODALITY=text` のように指定。ファイルと違うモダリティを指定した場合はエラーで停止
- transformer.py 単体は今までの設定のまま残し、改善前の基準として比較に使用
- train.py などの共通部分は変更なし

### 聞きたいこと

③のモダリティごとの分割について、ほかのモデル（GRU など）と合わせるルールが分からないため、教えていただきたいです。特に次の3点です。

- モダリティの分け方：どのモデルがどのモダリティを担当するか。自分の考えでは、Transformer は字幕（text）の担当がよいと思っています。これまでの結果で GRU との差が一番小さく（ρ：Transformer 0.369 / 0.307、GRU 0.424 / 0.361）、動画全体を一度に見られる自己注意が、文単位で値が入る字幕に合うと考えたためです
- 分け方の統一：ほかのモデルもモダリティごとにファイルを分けるのか（`gru_text.py` のような名前の付け方を含む）
- 渡し方：どのモダリティでどのモデルを使ったかを、ファインチューニング担当にどのような形で伝えるか（システム名・best.pt・コミットIDの一覧など）

### 学習について

学習はまだ行っていません。②の形をもう少し調整してから、映像・音声・字幕の3つを、シード0・1・2で学習する予定です。
