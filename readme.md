###readme file

##this repo will contain the project for the cv4e
readme.md

Markdown-formatted welcome file

Includes everything about:

1. Overview: what does this repo do?
2. Background: what is it useful for?
3. Installation instructions
4. Usage instructions | reproducing

results in a paper 5. Citation information

## installation instructions

```
conda env create -f cv4e.yml
conda activate cv4e
pip install setuptools==59.5.0
```

to test it works:

```
$ mv projects/urban_classifier/configs/cfg-test.yaml projects/urban_classifier/configs/cfg.yaml
$ python projects/urban_classifier/train.py
```
