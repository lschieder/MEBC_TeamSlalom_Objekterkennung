

# Prototyp: MEBC - TeamSlalom - Objekterkennung

Autor: Laurin Schieder

Version: 28.10.2025



### Neues Python-Environment anlegen und aktivieren

conda create --name yolo11-env python=3.12 -y
conda activate yolo11-env

### Ultralytics und PyTorch installieren

```bash
pip install ultralytics
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

```

Checken ob Pytorch GPU richtig installiet ist:

```bash
python -c "import torch; print(torch.cude.get_device_name(0))"
```

![](C:\Users\lauri\AppData\Roaming\marktext\images\2025-10-28-11-20-03-image.png)

### Daten sammeln & labeln

### Labels erstellen

Zb. Mit Labelstudio

### Datenstruktur vorbereiten

Quelle: `curl --output train_val_split.py https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/utils/train_val_split.py`




```bash
python train_val_split.py --datapath="C:\\Pfad\\zu\\deinen\\Daten" --train_pct=.8
```

### data.yaml für das Training anlegen

Öffne Texteditor, erstelle und passe diese Datei an:

```yml
path: C:\Users\<username>\Documents\yolo\data
train: train\images
val: validation\images
nc: 5
names: ["class1", "class2", "class3", "class4", "class5"]
```



### Training starten

```python
yolo detect train data=data.yaml model=yolo11s.pt epochs=60 imgsz=640
```

Das Modell wird dann als pt datei gespeichert bzw die weights unter `Objectdetection\runs\detect\train\weigts`





### Analyse der Ergebnise

![](C:\Users\lauri\AppData\Roaming\marktext\images\2025-10-28-16-35-20-confusion_matrix_normalized.png)

Die normalisierte Confusion Matrix zeigt die Leistung eines Klassifikationsmodells für drei Klassen (EndBoje, KursBoje, background), wobei EndBoje perfekt erkannt wird (1.00), KursBoje sehr gut klassifiziert wird (0.96), aber beide Bojen-Klassen zu 50% mit dem Hintergrund verwechselt werden. Das Modell hat Schwierigkeiten bei der Unterscheidung zwischen den Objektklassen und dem Hintergrund, während die Unterscheidung zwischen EndBoje und KursBoje nahezu fehlerfrei funktioniert.





![](C:\Users\lauri\AppData\Roaming\marktext\images\2025-10-28-16-38-10-results.png)

#### Trainingsverlauf

Die **Loss-Metriken** zeigen einen deutlichen Abwärtstrend über die gesamte Trainingsdauer. Die box_loss fällt von anfänglich 0.9 auf etwa 0.38, die cls_loss reduziert sich dramatisch von über 5.0 auf unter 0.5, und die dfl_loss sinkt von 1.15 auf etwa 0.80. Die Validierungs-Losses folgen einem ähnlichen Muster, wobei val/box_loss von 0.73 auf 0.51, val/cls_loss von 5.0 auf 0.5 und val/dfl_loss von 1.0 auf 0.88 abnehmen.

#### Leistungsmetriken

Die **Precision (B)** startet bei etwa 0.5 und steigt kontinuierlich auf nahezu 1.0 in den letzten Epochen, was bedeutet, dass fast alle vom Modell erkannten Objekte korrekt sind. Der **Recall (B)** beginnt bei sehr niedrigen 0.1 und erreicht ebenfalls Werte um 0.95-1.0, was zeigt, dass das Modell am Ende nahezu alle tatsächlich vorhandenen Objekte erkennt. Die **mAP50** (mean Average Precision bei IoU-Schwelle 0.5) steigt von 0.5 auf perfekte 1.0, während mAP50-95 (durchschnittlich über IoU-Schwellen 0.5-0.95) von 0.05 auf etwa 0.95 zunimmt, was auf eine sehr präzise Objektlokalisation hindeutet.

​

#### Konvergenz und Stabilität

Das Training zeigt nach etwa 40-50 Epochen eine deutliche **Konvergenz**, wobei die geglätteten Kurven (orange) flacher werden und die Metriken stabile Plateaus erreichen. Die starken Schwankungen in den frühen Epochen (besonders bei val/cls_loss und val/dfl_loss) deuten auf  initiale Anpassungsprozesse hin, die sich später stabilisieren. Die nahezu identischen Verläufe von Trainings- und Validierungsverlusten zeigen, dass kein signifikantes Overfitting vorliegt.
