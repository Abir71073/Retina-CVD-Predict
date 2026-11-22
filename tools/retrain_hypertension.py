"""Retrain the SAME EfficientNetB3 specialist with a corrected input contract.
Requires the original dataset; never run automatically or overwrite supplied weights.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--patient-column',required=True,help='Patient ID column: keeps both eyes in one partition')
    p.add_argument('--label-column',default='H')
    p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--output',type=Path,default=Path('retrained_hypertension'))
    a=p.parse_args()
    if a.output.exists():p.error('Use a new output directory to preserve previous training runs.')
    tf.keras.utils.set_random_seed(42)
    df=pd.read_csv(a.csv)
    required={'filename',a.patient_column,a.label_column}
    if not required.issubset(df.columns):p.error(f'Missing columns: {required-set(df.columns)}')
    if df.empty or df.filename.duplicated().any() or df[a.patient_column].isna().any():p.error('Require unique filenames and nonmissing patient IDs.')
    labels=pd.to_numeric(df[a.label_column],errors='coerce')
    if not labels.isin([0,1]).all():p.error('Training labels must be 0 or 1.')
    for name in df.filename:
        path=(a.images/str(name)).resolve()
        if not path.is_relative_to(a.images.resolve()) or not path.is_file():p.error(f'Invalid or missing image: {name}')
    df[a.label_column]=labels.astype(int).astype(str)
    splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42)
    train_idx,val_idx=next(splitter.split(df,groups=df[a.patient_column]))
    train_df,val_df=df.iloc[train_idx],df.iloc[val_idx]
    if any(part[a.label_column].nunique()!=2 for part in [train_df,val_df]):
        p.error('Both train and validation need both classes. Supply an adequately sized labelled dataset.')
    a.output.mkdir(parents=True)
    train_df.to_csv(a.output/'train_split.csv',index=False);val_df.to_csv(a.output/'validation_split.csv',index=False)
    # EfficientNet already contains Rescaling(1/255). Use raw pixels for NEW training.
    train_gen=ImageDataGenerator().flow_from_dataframe(train_df,directory=str(a.images),
        x_col='filename',y_col=a.label_column,classes=['0','1'],target_size=(300,300),
        batch_size=32,class_mode='binary',shuffle=True,seed=42,interpolation='nearest')
    val_gen=ImageDataGenerator().flow_from_dataframe(val_df,directory=str(a.images),
        x_col='filename',y_col=a.label_column,classes=['0','1'],target_size=(300,300),
        batch_size=32,class_mode='binary',shuffle=False,interpolation='nearest')
    base=tf.keras.applications.EfficientNetB3(weights='imagenet',include_top=False,input_shape=(300,300,3))
    base.trainable=True
    model=tf.keras.Sequential([base,tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dropout(.3),
                              tf.keras.layers.Dense(256,activation='relu'),
                              tf.keras.layers.Dense(1,activation='sigmoid',name='hypertension_output')])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),loss='binary_crossentropy',
                  metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])
    weights=compute_class_weight('balanced',classes=np.array([0,1]),y=train_gen.classes)
    model_path=a.output/'hypertension_specialist.keras'
    model.fit(train_gen,validation_data=val_gen,epochs=a.epochs,class_weight=dict(enumerate(weights)),
              callbacks=[tf.keras.callbacks.ModelCheckpoint(model_path,monitor='val_auc',mode='max',save_best_only=True),
                         tf.keras.callbacks.EarlyStopping(monitor='val_auc',mode='max',patience=5,restore_best_weights=True),
                         tf.keras.callbacks.CSVLogger(a.output/'training_history.csv')])
    manifest={'file':model_path.name,'preprocessing':'rgb_0_255','class_indices':train_gen.class_indices,
              'positive_label':f'{a.label_column}=1','patient_column':a.patient_column,
              'train_count':len(train_df),'validation_count':len(val_df),
              'note':'Evaluate on a separate held-out patient set before replacing any deployed checkpoint.'}
    (a.output/'model_metadata.json').write_text(json.dumps(manifest,indent=2))
    print('Saved new weights and metadata to',a.output)
    print('Existing weights and model_config.json were not modified. Replace BOTH together after evaluation.')

if __name__=='__main__':main()
