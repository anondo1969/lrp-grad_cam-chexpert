import numpy as np
import os
from configparser import ConfigParser
from generator import AugmentedImageSequence
from models.keras import ModelFactory
from sklearn.metrics import roc_auc_score, average_precision_score
from utility import get_sample_counts
import pickle


def main():
    # parser config
    config_file = "./sample_config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DEFAULT"].get("output_dir")
    base_model_name = cp["DEFAULT"].get("base_model_name")
    class_names = cp["DEFAULT"].get("class_names").split(",")
    image_source_dir = cp["DEFAULT"].get("image_source_dir")

    # train config
    image_dimension = cp["TRAIN"].getint("image_dimension")

    # test config
    batch_size = cp["TEST"].getint("batch_size")
    test_steps = cp["TEST"].get("test_steps")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")

    # parse weights file path
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(output_dir, output_weights_name)
    best_weights_path = os.path.join(output_dir, "best_{output_weights_name}")

    # get test sample count
    test_counts, _ = get_sample_counts(output_dir, "valid", class_names)

    # compute steps
    if test_steps == "auto":
        test_steps = int(test_counts / batch_size)
    else:
        try:
            test_steps = int(test_steps)
        except ValueError:
            raise ValueError("""
                test_steps: {test_steps} is invalid,
                please use 'auto' or integer.
                """)
    print("** test_steps: {test_steps} **")

    print("** load model **")
    if use_best_weights:
        print("** use best weights **")
        model_weights_path = best_weights_path
    else:
        print("** use last weights **")
        model_weights_path = weights_path
    model_factory = ModelFactory()
    model = model_factory.get_model(
        class_names,
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=model_weights_path, input_shape=(image_dimension, image_dimension, 3))

    print("** load test generator **")
    test_sequence = AugmentedImageSequence(
        dataset_csv_file=os.path.join(output_dir, "valid.csv"),
        class_names=class_names,
        source_image_dir=image_source_dir,
        batch_size=batch_size,
        target_size=(image_dimension, image_dimension),
        augmenter=None,
        steps=test_steps,
        shuffle_on_epoch_end=False,
    )

    print("** make prediction **")
    
    
    y_hat = model.predict_generator(test_sequence, verbose=1)
    y = test_sequence.get_y_true()
    
    results_dict = dict()
    results_dict['y_predicted'] = y_hat
    results_dict['y_true'] = y
    
    with open("results_dict", "wb") as fp:
        pickle.dump(results_dict, fp)
    print("results_dict saved")

    test_log_path = os.path.join(output_dir, "results.txt")
    aurocs = []
    
    print("class: AUROC score")
    
    for i in range(len(class_names)):
        try:
            score = roc_auc_score(y[:, i], y_hat[:, i])
                
        except ValueError:
            score = np.nan
        aurocs.append(score)
            
        print(str(class_names[i])+": {:.2f}".format(score))
    
    #Overall evaluation without 'Fracture' class
    y_score = np.delete(y_hat, 12, 1)
    y_true = np.delete(y, 12, 1)
    
    print("weighted average AUROC: {:.2f} ".format(roc_auc_score(y_true, y_score, average='weighted')))
    print("weighted average AUPRC: {:.2f} ".format(average_precision_score(y_true, y_score, average='weighted')))
    


if __name__ == "__main__":
    main()
