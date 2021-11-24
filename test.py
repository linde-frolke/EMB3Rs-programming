from tests.test_lib import defineArguments, processInput

# Write Here all the available tests you want to run
availableTests = {
    "example:test": "fuction_name",
}


def init():
    # DO NOT CHANGE FROM THIS POINT BELOW
    # UNLESS YOU KNOW WHAT YOUR DOING
    args = defineArguments(availableTests)

    processInput(args, availableTests)
