# coco ann file(ex. instances_train2017.json) -> class filtered images,label(yolov5 format. .txt)


# How To

1. Convert origin ann file to filtered ann file
```python
python filter.py --input_json ./instnaces_train2017.json --output_json ./filtered_train2017.json --categories person cat dog
```

2. Download images and extract labels
```python
python coco2yolo.py --input_json ./filtered_train2017.json --image_output_dir ./images/filtered__train2017 --label_output_dir ./labels/filtered__train2017
```


## References

1. https://github.com/immersive-limit/coco-manager
2. https://blog.naver.com/ehdrndd/222708300652