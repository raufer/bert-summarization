#### Download the CNN/DM Processed Data
```
mkdir -p data
aws s3 cp --recursive s3://waymark-experiments/summarization/data data/

mkdir data/cnn-dm/chunked/train
mkdir data/cnn-dm/chunked/test
mkdir data/cnn-dm/chunked/val

mv data/cnn-dm/chunked/train_* data/cnn-dm/chunked/train/
mv data/cnn-dm/chunked/test_* data/cnn-dm/chunked/test/
mv data/cnn-dm/chunked/val_* data/cnn-dm/chunked/val/
````