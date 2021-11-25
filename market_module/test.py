from .tests.test_lib import defineArguments, processInput
from .short_term.tests.test_prototype_p2p import test_p2p
from .short_term.tests.test_prototype_pool import test_pool

# Write Here all the available tests you want to run
availableTests = {
    "test:shortterm:p2p": test_p2p,
    "test_short_term1": test_pool
}


def init():
    # DO NOT CHANGE FROM THIS POINT BELOW
    # UNLESS YOU KNOW WHAT YOUR DOING
    args = defineArguments(availableTests)

    processInput(args, availableTests)
