from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import classification_report

from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt
import csv

import numpy as np
import os
from configparser import ConfigParser
from generator import AugmentedImageSequence
from models.keras import ModelFactory
from sklearn.metrics import roc_auc_score
from utility import get_sample_counts
import pandas as pd

import innvestigate
import innvestigate.utils

#CUDA_VISIBLE_DEVICES=0,1 python3 lrp_test_visualization.py &>> lrp.alpha_2_beta_1_after_157.txt &

def transparent_cmap(cmap, N=255):
  "Copy colormap and set alpha values"

  mycmap = cmap
  mycmap._init()
  mycmap._lut[:,-1] = np.linspace(0, 0.4, N+4)
  return mycmap

def plot_LRP(heatmap, x1):
  w, h = heatmap.shape
  y, x = np.mgrid[0:h, 0:w]   
  mycmap = transparent_cmap(plt.cm.Reds)
  fig, ax = plt.subplots(1, 1)
  ax.imshow(x1, cmap='gray')
  cb = ax.contourf(x, y, heatmap, 2, cmap=mycmap)
  plt.colorbar(cb)
  plt.savefig("testtt.png")

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
    

    test_data_df = pd.read_csv(os.path.join(output_dir, "valid.csv"))
    
    np_names = np.array(class_names)
    
    lrp_methods = ["lrp.z", "lrp.epsilon", "lrp.alpha_2_beta_1", "deep_taylor"]
            
    # Strip softmax layer
    #model = innvestigate.utils.model_wo_softmax(model)
    #print("removed softmax")
    best_count = 0
    
    for lrp_method in lrp_methods:
        
    
        # Create analyzer
        if lrp_method=="lrp.epsilon":
            optional_method = {'epsilon': 1}
        else:
            optional_method = {}
            
        '''
        {'input': <class 'innvestigate.analyzer.misc.Input'>, 
            'random': <class 'innvestigate.analyzer.misc.Random'>, 
            'gradient': <class 'innvestigate.analyzer.gradient_based.Gradient'>, 
            'gradient.baseline': <class 'innvestigate.analyzer.gradient_based.BaselineGradient'>, 
            'input_t_gradient': <class 'innvestigate.analyzer.gradient_based.InputTimesGradient'>, 
            'deconvnet': <class 'innvestigate.analyzer.gradient_based.Deconvnet'>, 
            'guided_backprop': <class 'innvestigate.analyzer.gradient_based.GuidedBackprop'>, 
            'integrated_gradients': <class 'innvestigate.analyzer.gradient_based.IntegratedGradients'>, 
            'smoothgrad': <class 'innvestigate.analyzer.gradient_based.SmoothGrad'>, 
            'lrp': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRP'>, 
            'lrp.z': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPZ'>, 
            'lrp.z_IB': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPZIgnoreBias'>, 
            'lrp.epsilon': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPEpsilon'>, 
            'lrp.epsilon_IB': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPEpsilonIgnoreBias'>, 
            'lrp.w_square': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPWSquare'>, 
            'lrp.flat': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPFlat'>, 
            'lrp.alpha_beta': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlphaBeta'>, 
            'lrp.alpha_2_beta_1': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlpha2Beta1'>, 
            'lrp.alpha_2_beta_1_IB': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlpha2Beta1IgnoreBias'>, 
            'lrp.alpha_1_beta_0': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlpha1Beta0'>, 
            'lrp.alpha_1_beta_0_IB': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlpha1Beta0IgnoreBias'>, 
            'lrp.z_plus': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPZPlus'>, 
            'lrp.z_plus_fast': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPZPlusFast'>, 
            'lrp.sequential_preset_a': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPSequentialPresetA'>, 
            'lrp.sequential_preset_b': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPSequentialPresetB'>, 
            'lrp.sequential_preset_a_flat': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPSequentialPresetAFlat'>, 
            'lrp.sequential_preset_b_flat': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPSequentialPresetBFlat'>, 
            'deep_taylor': <class 'innvestigate.analyzer.deeptaylor.DeepTaylor'>, 
            'deep_taylor.bounded': <class 'innvestigate.analyzer.deeptaylor.BoundedDeepTaylor'>, 
            'deep_lift.wrapper': <class 'innvestigate.analyzer.deeplift.DeepLIFTWrapper'>, 
            'pattern.net': <class 'innvestigate.analyzer.pattern_based.PatternNet'>, 
            'pattern.attribution': <class 'innvestigate.analyzer.pattern_based.PatternAttribution'>}
        '''
            
        analyzer = innvestigate.create_analyzer(lrp_method, model, neuron_selection_mode="index", **optional_method)
            
        best_count = 0
        for i in range(len(test_data_df)):
            
            best_count+=1
            
            if best_count>0:
            
        
                image_name= image_source_dir+'/'+test_data_df['Path'][i]
                image = Image.open(image_name)
                image_array = np.asarray(image.convert("RGB"))
                image_array = image_array / 255.
                image_array = resize(image_array, (image_dimension, image_dimension))
                image_array = image_array[None, :, :, :]
        

                #ind = y_hat[j].argsort()[-4:][::-1]
            
                best_ind = y_hat[i].argsort()[-1:][::-1]
            
                ind = y_hat[i].argsort()[::-1]
        
                for output_neuron in ind:
            
                    class_name = np_names[output_neuron]
            
                    # Apply analyzer w.r.t. maximum activated output-neuron
                    a = analyzer.analyze(image_array, neuron_selection=output_neuron)

            

                    # Aggregate along color channels and normalize to [-1, 1]
                    a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
                    a /= np.max(np.abs(a))

                    # Plot

                    #plot_LRP(a[0], image_array[0])
            
                    plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
                
                
                    plt.savefig('all_predictions/'+lrp_method+'/'+test_data_df['Path'][i].replace("/","_")+'_heatmap_using_lrp_method_'+lrp_method+'_class_name_'+class_name+'.png')
                
                    if output_neuron==best_ind:
                    
                        print("Best count: " + str(best_count))
                        plt.savefig('best_predictions/'+lrp_method+'/'+test_data_df['Path'][i].replace("/","_")+'_heatmap_using_lrp_method_'+lrp_method+'_best_class_name_'+class_name+'.png')

    

if __name__ == "__main__":
    main()

