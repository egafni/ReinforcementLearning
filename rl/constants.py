import os

opj = os.path.join

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ex: /home/spock/projects/TheOracle
ORACLE_DIR = opj(REPO_DIR, "algos")  # ex: /home/spock/projects/TheOracle/algos
TESTS_DIR = opj(ORACLE_DIR, "tests")  # ex: /home/spock/projects/TheOracle/algos/tests
SCHEDULED_SCRIPTS_DIR = opj(SCRIPTS_DIR, "scheduled")  # ex: /home/spock/projects/TheOracle/scripts/scheduled

STATIC_DIR = "/static"

