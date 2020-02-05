# This module profiles the memory usage of a proto cloud function.
import pickle
from memory_profiler import profile


LOG_FILE = "./log/memory_proto_function.log"


def load_decoder():
    path = "../model/experiment_01_decoder.pkl"
    with open(path, "rb") as model:
        return pickle.load(model)


def load_model():
    path = "../model/experiment_01_model.pkl"
    with open(path, "rb") as model:
        return pickle.load(model)


@profile(stream=open(LOG_FILE,"w+"))
def run(data):
    model = load_model()
    decoder = load_decoder()
    predictions = model.predict(data)
    return [decoder[i] for i in predictions]


if __name__ == "__main__":
    # For the test case I chose a user id that is out of sample and sensor data
    # zeroed out. Two test cases are used but it could have been any number of
    # cases.
    next_user = len(load_decoder()) + 1

    data = [
        [next_user, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [next_user, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]

    results = run(data)

    print("=" * 80)
    print(f"Inputs: {data}")
    print(f"Predections: {results}")
    print("=" * 80)

# Returns
# ==============================================================================
# Inputs: [[7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
# Predections: ['stairsdown', 'stairsdown']
# ==============================================================================
