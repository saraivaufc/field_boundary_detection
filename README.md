# field_segmentation
field_segmentation

## Create virtualenv

```shell
$ python3 -m venv env
$ source env/bin/activate
```

## Install requirements

```shell
$ pip3 install -r requirements.txt
```

## Install SAM

```shell
$ pip3 install git+https://github.com/facebookresearch/segment-anything.git
```

## Install GDAL

```shell
$ sudo apt-get update
$ sudo apt-get install libgdal-dev
$ sudo apt-get install python3-dev
$ sudo apt-get install gdal-bin python3-gdal
$ pip3 install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"
```

## Download SAM

```shell
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```