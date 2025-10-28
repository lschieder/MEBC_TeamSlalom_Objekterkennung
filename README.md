





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

### Modell-Inferenz (auf Webcam, Bildern oder Videos ausführen)

curl -o yolo_detect.py https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/yolo_detect.py
python yolo_detect.py --model=runs/detect/train/weights/best.pt --source=usb0

### Beispiel für Video-Inferenz mit festgelegter Auflösung:

python yolo_detect.py --model=yolo11s.pt --source=test_vid.mp4 resolution=1280x720

### Fertig! Modelle und Trainingsgraphen findest du unter: runs/detect/train/
