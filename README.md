# Image2Event

<img src="Taxi.Driver.1976.HR-HDTV.jpg" alt="BGM" style="zoom: 33%;" />

Towards unsupervised domain adaptation on event data via labeled image data.

## Training

```python
python train.py
```

Default network setting is res2net (or resnet18) as Backbone, FPN as Detection Head, FCN as Upsampling & Downsampling Decoder.

## Inference

```python
python train.py -e
```

Results stored in the "Inference" directory.

## Using TensorBoard

To visualize training progress with TensorBoard, use the following command:

```python
tensorboard --logdir ./checkpoints/log/
```