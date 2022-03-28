import pytest
import ray


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def ray_cluster():
    # https://docs.ray.io/en/master/auto_examples/testing-tips.html#tip-2-use-ray-init-local-mode-true-if-possible
    # object_store_memory is how much of /dev/shm (shared ram) to reserve
    # local_mode = True runs Ray in a single process
    ray.init(local_mode=True, num_cpus=1, num_gpus=0, object_store_memory=100e6)
    yield
    ray.shutdown()
