# narrative-event-detection
 A Multilingual Dataset and Classifier for Narrative Events

 The data used in the KFold classifiers can be found under the folder data. 

 Folder KFold_code:
 - train_event_classifier_crf.py: KFold prediction
 - correct_oof_predicitions.py: correct inconsistent BIO tags of Kfold prediction
 - evaluate_oof_corrected.ipynb: evaluated corrected data

 Folder external:
 - train_final_fit.py: train the KFold train models over all datat for a final 
 - predict_and_evaluate_external.py: predict on unseen data
 - correct_external_predictions.py: correct inconsistent BIO tags on unseen data
 - evaluate_external_corrected.ipynb: evaluate corrected unseen data
 - metamorphosis: annotated Metamorphosis data

Folder narrativity arcs:
- Metamorphosis_German.csv: German original text gold standard data to compare other languages to
- narrative_graph.ipynb: Create narrativity graphs

 
