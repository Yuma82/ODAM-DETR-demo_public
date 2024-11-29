# ODAM-DETR-demo

DETRモデルの判断根拠をODAMによって可視化するコードが公開されておらず（アクセス権限がない状態），手っ取り早く動作させるために，DETRのデモモデルを用いて判断根拠をODAMで可視化するコードを作成した．
DETRのデモコードは[公式実装](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb)を参考にした．

追記：ODAMの作者に問い合わせたところ，Colabノートブックを公開していただいた．
公式のデモ実装は[こちら](https://colab.research.google.com/drive/1j-JZuZ3FXXQucr_LZWSxCUmHsqpWzaVR#scrollTo=h91rsIPl7tVl)．

---

# DETRのdemoコードからの修正点

### **出力の追加**

- **`conv_features` の追加**:
    
    **オリジナルのコード**: 
    `return` の部分で出力されるのは `'pred_logits'`（クラスラベルの予測）と `'pred_boxes'`（バウンディングボックスの予測）のみ．
    
    **修正点**:
    これらに加えて，`'conv_features'` が出力されるように変更．この `conv_features` は，ResNet-50 のバックボーンから得られた特徴量（`x`）をそのまま追加している．
    

### **画像入力処理の拡張**

- **単一画像から複数画像への対応**:
    
    **オリジナルのコード**:
    特定のURLから画像を取得し，単一の画像に対して `detect` 関数を適用する形式．
    
    **修正点**:
    ローカルディレクトリ内の複数の画像ファイルを処理対象とするように変更．
    
    - サポートされる画像形式（`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`）を指定し，指定したディレクトリ内の画像をすべてリストアップ．
    - この変更により，大量のデータセットに対してバッチ処理が可能となった．
    
    **使用例**:
    
    ```python
    directory_path = "/path/to/dataset"
    image_paths = [
        os.path.join(directory_path, file)
        for file in os.listdir(directory_path)
        if os.path.splitext(file)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    ]
    for image_path in image_paths:
        im = Image.open(image_path)
        scores, boxes = detect(im, detr, transform)
    ```
    

---# ODAM-DETR
