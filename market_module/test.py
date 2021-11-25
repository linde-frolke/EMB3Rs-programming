from .tests.test_lib import defineArguments, processInput
from .short_term.tests.test_prototype_p2p import test

# Write Here all the available tests you want to run
availableTests = {
    "example:test": test,
}


def init():
    # DO NOT CHANGE FROM THIS POINT BELOW
    # UNLESS YOU KNOW WHAT YOUR DOING
    args = defineArguments(availableTests)

    processInput(args, availableTests)
