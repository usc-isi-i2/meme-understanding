import imp
from src.datasets.mami import output_keys

def calculate(pred, output, correct_count_map):
    actual_output = []
    for i in range(len(output[output_keys[0]])):
        sample_output = []
        for j, output_key in enumerate(output_keys):
            sample_output.append(int(output[output_key][i]))
        actual_output.append(sample_output)

    for i, sample_pred in enumerate((pred > 0.5).int().tolist()):
        for j , output_key in enumerate(output_keys):
            correct_count_map[output_key] += 1 if sample_pred[j] == actual_output[i][j] else 0

    return actual_output
