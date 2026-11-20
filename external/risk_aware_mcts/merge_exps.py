import sys
import glob
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main(data_folder_path=None):

    print("Merging exp. files...")

    f_vals_list = []
    for filename in glob.iglob(data_folder_path + '**/*.json', recursive=True):
        print(filename)

        if filename == data_folder_path + "exp_data.json":
            continue

        with open(filename, 'r') as f:
            data = json.load(f)
            data = json.loads(data)
        f.close()
        f_vals_list.extend(data["f_vals"])

    merged_dict = data
    merged_dict["f_vals"] = f_vals_list
    # print(merged_dict)
    print("Number of experiments:", len(merged_dict["f_vals"]))

    # Dump merged dict.
    f = open(data_folder_path + "/exp_data.json", "w")
    dumped = json.dumps(merged_dict, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

if __name__ == "__main__":
    main(sys.argv[1])
