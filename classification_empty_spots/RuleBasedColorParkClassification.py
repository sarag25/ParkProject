import os
from collections import defaultdict
from RuleBasedColorParkClassificationFunctions import classify_parking_slot, show_example_with_mask_and_hist, \
    FOLDER_TO_LABEL

'''
Given an image of an empty slot classifies its type according to the color of the lines
'''

example_images = {}

def evaluate_dataset(root_dir):
    """
    Evaluates dataset_emptyslots using functions defined in RuleBasedColorParkClassificationFunctions.
    """
    correct = 0
    total = 0
    results = []

    class_stats = defaultdict(lambda: {
        "tp": 0, "fp": 0, "fn": 0, "total": 0
    })

    labels = list(FOLDER_TO_LABEL.values())
    confusion_matrix = {label: defaultdict(int) for label in labels}

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        expected_label = FOLDER_TO_LABEL.get(folder)
        if expected_label is None:
            print(f"Skipping unknown folder: {folder}")
            continue

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(folder_path, filename)
            try:
                predicted_label = classify_parking_slot(image_path, visualize=False)
                total += 1
                is_correct = (predicted_label == expected_label)
                correct += int(is_correct)
                class_stats[expected_label]["total"] += 1
                if predicted_label == expected_label:
                    class_stats[expected_label]["tp"] += 1
                else:
                    class_stats[expected_label]["fn"] += 1
                    class_stats[predicted_label]["fp"] += 1

                confusion_matrix[expected_label][predicted_label] += 1

                results.append((image_path, expected_label, predicted_label, is_correct))
                print(f"{filename}: Predicted={predicted_label}, Expected={expected_label}, {'[Correct]' if is_correct else '[Error]'}")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    print(f"\nCorrect: {correct}/{total} ({(correct/total)*100:.2f}%)\n")

    for label in labels:
        stats = class_stats[label]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        total_cls = stats["total"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = tp / total_cls if total_cls > 0 else 0

        print(f"Class: {label}")
        print(f"  Accuracy : {accuracy:.2f}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall   : {recall:.2f}")
        print(f"  F1-Score : {f1:.2f}")
        print()

    print("Confusion Matrix:")
    col_width = 12
    header = "Actual\\Pred".ljust(col_width) + "".join(label.ljust(col_width) for label in labels)
    print(header)
    for actual in labels:
        row = actual.ljust(col_width)
        for predicted in labels:
            count = str(confusion_matrix[actual][predicted])
            row += count.ljust(col_width)
        print(row)

    return results


results = evaluate_dataset("../datasets/dataset_emptyslots")
show_example_with_mask_and_hist(results)