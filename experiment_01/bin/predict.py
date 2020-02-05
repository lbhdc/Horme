# This program is an interface to test the model produced from experiment 01.
# This model requires two pieces of data before it is able to make a prediction,
# the users id, and the users accelerometer and gyroscope sensor readings. In
# this case the model is only looking at a single instance of sensor data, but
# it could also predict batches.
#
# Examples:
# $ python predict.py -id 0 --sensor 0.0 0.0 0.0 0.0 0.0 0.0
# ==============================================================================
# Inputs: [[0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
# Predictions: ['stairsup']
# ==============================================================================
import argparse
import pickle


def load_decoder():
    path = "../model/experiment_01_decoder.pkl"
    with open(path, "rb") as model:
        return pickle.load(model)


def load_model():
    path = "../model/experiment_01_model.pkl"
    with open(path, "rb") as model:
        return pickle.load(model)


def run(data):
    # Run is the proto version of the future cloud function this will become. It
    # takes a matrix like data structure that can represent many sensor
    # readings. A list of predictions is returned.
    model = load_model()
    decoder = load_decoder()
    predictions = model.predict(data)
    return [decoder[i] for i in predictions]


def cli_args():
    parser = argparse.ArgumentParser(description="Experiment 01 Model")

    parser.add_argument(
        "-id",
        type=int,
        default=0,
        help="<user_id>"
    )

    parser.add_argument(
        "-s", "--sensor",
        nargs=6,
        type=float,
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        help="<accel_x> <accel_y> <accel_z> <gyro_x> <gyro_y> <gyro_z>"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = cli_args()
    data = [[args.id, *args.sensor]]
    results = run(data)

    print("=" * 80)
    print(f"Inputs: {data}")
    print(f"Predictions: {results}")
    print("=" * 80)
