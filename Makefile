.PHONY: data train eval inspect

data:
	python data/prepare.py

inspect:
	python data/inspect.py

train:
	python train.py --config configs/train.yaml

eval:
	python eval/run_eval.py
