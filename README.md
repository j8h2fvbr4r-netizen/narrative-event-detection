# narrative-event-detection
 A Multilingual Dataset and Classifier for Narrative Events

 The data used in the KFold classifiers can be found under the folder data.
 The following columns are shown in each .csv file:
 - source_file: filename of original text
 - annotator: should always be CURATION_USER for the gold standard dataset
 - sentence_id: sentence number that token is in
 - sentence_text: sentence that token is in
 - token: token that the event_bio and pmode_bio labels belong to
 - start: character offset beginning token
 - stop: character offset end of token
 - event_bio: event label
 - event_bio_2: event label discontinuous span
 - pmode_bio: presentation mode 
 - pmode_bio_2: presentation mode discontinuous span


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

 
