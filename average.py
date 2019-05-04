def average(predictions):
    for pred in predictions:
        print(pred.out)
    return predictions[0].out

def hard_average(predictions):
    # get the length of each prediction
    pred_len = len(predictions[0].out)
    # initialize a list of counters
    counters = [0 for i in range(pred_len)]

    for pred in predictions:
        # increase the counter of highest probability in the prediction
        max_place = pred.out.max(0)[1]
        counters[max_place] += 1

    # return the confidences with 1 in the maximum place and 0 in the others
    max_place = counters.index(max(counters))
    out = [0 for i in range(pred_len)]
    out[max_place] = 1
    return out


def soft_average(predictions):
    # sum all predictions
    out = sum([pred.out for pred in predictions])
    # evaluate the average
    out /= len(predictions)
    return out

def depth_average(predictions):
    # sum all predictions
    out = sum([pred.out * pred.bounding_box.area() for pred in predictions])
    # evaluate the average
    out /= sum(out)
    return out

def depth_sqrt_average(predictions):
    # sum all predictions
    out = sum([pred.out * (pred.bounding_box.area()**.5) for pred in predictions])
    # evaluate the average
    out /= sum(out)
    return out