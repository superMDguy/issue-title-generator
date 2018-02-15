clean-data:
	rm -rf ./data
	mkdir data

datagen: clean-data
	 t2t-datagen --t2t_usr_dir=./issues_problem --data_dir=./data --problem=issue_to_title

.PHONY: clean-data datagen
