clean-data:
	rm -rf ./data
	mkdir data

datagen: clean-data
	cp ~/Code/dl/datasets/github_issues.csv ~/github_issues.csv # put on SSD to speed up
	TF_CPP_MIN_LOG_LEVEL=3 t2t-datagen --t2t_usr_dir=./issues_problem --data_dir=./data --problem=issue_to_title
	rm ~/github_issues.csv

make clean-train:
	rm -rf ./train

train:
	t2t-trainer \
	--t2t_usr_dir=./issues_problem \
	--data_dir=./data \
	--problem=issue_to_title \
	--model=transformer \
	--hparams_set=transformer_prepend \
	--hparams='batch_size=1024' \
	--output_dir=./train \
	--eval_steps=500

tensorboard:
	sudo tensorboard --logdir ./train --host 0.0.0.0 --port 80

.PHONY: clean-data datagen train tensorboard
