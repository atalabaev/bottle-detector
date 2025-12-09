#  Bottle Detector (YOLOv8)

Учебный проект по компьютерному зрению: детекция бутылок на изображениях и в реальном времени с веб-камеры с использованием **YOLOv8**.

Проект показывает полный ML-цикл:
> датасет → обучение → инференс → аналитика.


## Dataset

Датасет хранится отдельно и доступен по ссылке:

https://drive.google.com/file/d/1R3vZ6hUXyu7PUPbZMe5ifn2F-YubBeB6/view?usp=sharing

После распаковки структура должна быть:
out/data/images/{train,val,test}
out/data/labels/{train,val,test}

##  Установка

```bash
git clone https://github.com/atalabaev/bottle-detector.git
cd bottle-detector

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

 Обучение
python3 src/train_yolo.py --data out/data/bottle.yaml --model yolov8s.pt --epochs 30 --batch 8
После обучения:
cp runs/train/<run_name>/weights/best.pt models/best.pt
 Детекция с камеры (real-time)
python3 camera_infer_improved.py --weights models/best.pt --camera 0 --conf 0.05
Клавиши:
•	s — сохранить кадр
•	q / Esc — выход
Все детекции сохраняются в detection_log.csv.
