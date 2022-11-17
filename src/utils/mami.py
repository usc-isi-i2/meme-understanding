import torch as t

# Find a better name of function /or/ divide the logic in two separate functions
def calculate(pred, output, actual_labels, pred_labels, output_keys):
    actual_output = []
    for i in range(len(output[output_keys[0]])):
        sample_output = []
        for j, output_key in enumerate(output_keys):
            sample_output.append(int(output[output_key][i]))
        actual_output.append(sample_output)

    pred_sogmoid = t.sigmoid(pred)
    for i, sample_pred in enumerate((pred_sogmoid > 0.5).int().tolist()):
        for j , output_key in enumerate(output_keys):
            actual_labels[output_key].append(actual_output[i][j])
            pred_labels[output_key].append(sample_pred[j])

    return actual_output
