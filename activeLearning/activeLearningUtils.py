from math import radians, sin, cos, sqrt, atan2

def calculate_distance(coord1, coord2) -> float:
    'Distance betweens two points on earth according to the Haversine formula'

    # Function to calculate distance between two coordinates
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6371 * c  # Radius of Earth in kilometers
    return distance

def compute_precision_recall_f1(instances, _truth, _pred):
    'Computation algorithm for precision, recall and f1-score'
    
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    matched_coordinates = []

    for instance in instances:
        ground_truth = instance[_truth]
        predicted = instance[_pred]
        matched_ground_truth = set()  # To keep track of matched ground truth elements
        
        true_positives = 0
        for pred_key, pred_coord in predicted.items():
            matched = False
            for gt_key, gt_coord in ground_truth.items():
                if pred_key.lower() in gt_key.lower() or gt_key.lower() in pred_key.lower():
                    if gt_key not in matched_ground_truth:  # Ensure we don't count the same ground truth multiple times
                        true_positives += 1
                        matched_ground_truth.add(gt_key)
                        matched_coordinates.append((pred_coord, gt_coord))
                        matched = True
                        break
            
            # False positives are elements in predicted that did not match any ground truth element
            if not matched:
                total_false_positives += 1
        
        # False negatives are ground truth elements that did not match any predicted element
        total_false_negatives += len(ground_truth) - len(matched_ground_truth)
        total_true_positives += true_positives
    
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score, matched_coordinates

def calculate_A_at_k(matched_coordinates, k):
    'Calculate accuracy of coordinates in Toleranceradius k'

    # Function to calculate accuracy at k (A@k)
    correct_matches = 0
    for pred_coord, truth_coord in matched_coordinates:
        if calculate_distance(pred_coord, truth_coord) <= k:
            correct_matches += 1

    accuracy_at_k = (correct_matches / len(matched_coordinates)) * 100 if matched_coordinates else 0
    return accuracy_at_k

    # MLFlow Tracking
    # with mlflow.start_run(
    #     run_name="Retrain-Job",
    #     tags={"job": "retrain"},
    #     description="Retrain-Job of configured model in provider"
    # ):
        # Speichern der Metriken in MLFlow
        # mlflow.log_metric("precision", eval_results['eval_precision'])
        # mlflow.log_metric("recall", eval_results['eval_recall'])
        # mlflow.log_metric("f1", eval_results['eval_f1'])

        # Speichern der Trainingsparameter
        # mlflow.log_param("learning_rate", training_args.learning_rate)
        # mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
        # mlflow.log_param("num_train_epochs", training_args.num_train_epochs)