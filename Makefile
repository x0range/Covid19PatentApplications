raw_data: src/download_raw_data.py
	mkdir -p data/raw
	pipenv run python src/download_raw_data.py -y 2014,2015,2016,2017,2018,2019,2020
	pipenv run python src/parse_raw_data.py
