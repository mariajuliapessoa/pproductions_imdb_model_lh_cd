install:
	pip install -r requirements.txt

lint:
	flake8 lh_cd_mariajuliapessoa/

run-notebook:
	jupyter notebook notebooks/
