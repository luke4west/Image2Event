# Image2Event

<img src="Taxi.Driver.1976.HR-HDTV.jpg" alt="BGM" style="zoom: 33%;" />

Towards unsupervised domain adaptation on event data.

## Training

```python
python train.py
```

Default network setting is res2net as Backbone, FPN as Detection Head, FCN as Upsampling & Downsampling Decoder.

## Inference

```python
python train.py -e
```

Detection results will be stored in the "Inference" directory.

## Using TensorBoard

```python
tensorboard --logdir ./checkpoints/log/
```

To visualize training progress with TensorBoard, use the above command
