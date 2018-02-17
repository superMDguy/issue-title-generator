clean-data:
	rm -rf ./data
	mkdir data

datagen: clean-data
	cp ~/Code/dl/datasets/github_issues.csv ~/github_issues.csv # put on SSD to speed up
	TF_CPP_MIN_LOG_LEVEL=3 t2t-datagen --t2t_usr_dir=./issues_problem --data_dir=./data --problem=issue_to_title
	rm ~/github_issues.csv

train:
	t2t-trainer \
  --t2t_usr_dir=./issues_problem \
	--data_dir=./data \
  --problems=issue_to_title \
  --model=transformer \
  --hparams_set=transformer_prepend \
  --output_dir=./train

.PHONY: clean-data datagen train
