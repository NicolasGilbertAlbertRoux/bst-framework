.PHONY: check dry-run closure projection memory arbitration zeta geometry strings robustness

check:
	python scripts/check_repository.py

dry-run:
	python scripts/reproduce_minimal.py --dry-run

closure:
	python scripts/reproduce_minimal.py --run --category 01_closure --timeout 120 --keep-going

projection:
	python scripts/reproduce_minimal.py --run --category 02_projection --timeout 120 --keep-going

memory:
	python scripts/reproduce_minimal.py --run --category 03_memory --timeout 120 --keep-going

arbitration:
	python scripts/reproduce_minimal.py --run --category 04_arbitration --timeout 120 --keep-going

zeta:
	python scripts/reproduce_minimal.py --run --category 05_zeta --timeout 120 --keep-going

geometry:
	python scripts/reproduce_minimal.py --run --category 06_geometry --timeout 120 --keep-going

strings:
	python scripts/reproduce_minimal.py --run --category 07_string_bridge --timeout 120 --keep-going

robustness:
	python scripts/reproduce_minimal.py --run --category 08_robustness --timeout 120 --keep-going
