import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- 1. CONFIGURATION ---
from pathlib import Path
from PIL import Image
DRIVE_TRAIN_DIR = os.environ.get('DRIVE_TRAIN_DIR', str(Path(__file__).resolve().parents[1] / 'drive_dataset' / 'training')) 
PATCH_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 30
STEPS_PER_EPOCH = 200 # Defines how many batches constitute one epoch

# --- 2. U-NET ARCHITECTURE ---
def build_unet(input_shape=(PATCH_SIZE, PATCH_SIZE, 3)):
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)
    
    c2 = Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)
    
    c3 = Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)
    
    # Bottleneck
    c4 = Conv2D(256, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(256, 3, activation='relu', padding='same')(c4)
    
    # Decoder
    u5 = UpSampling2D()(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(128, 3, activation='relu', padding='same')(u5)
    c5 = Conv2D(128, 3, activation='relu', padding='same')(c5)
    
    u6 = UpSampling2D()(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(64, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(64, 3, activation='relu', padding='same')(c6)
    
    u7 = UpSampling2D()(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(32, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(32, 3, activation='relu', padding='same')(c7)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(c7)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 3. MEMORY-SAFE ON-THE-FLY PATCH GENERATOR ---
class DRIVEDatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_dir, mask_dir, batch_size=16, patch_size=256, steps_per_epoch=200):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.steps_per_epoch = steps_per_epoch
        
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
        self.images = []
        self.masks = []
        
        print("Pre-loading 20 base images into memory...")
        for filename in self.image_files:
            base_name = filename.split('_')[0]
            
            # Load and normalize image
            img_path = os.path.join(img_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f'Unreadable training image: {img_path}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            # Load and normalize mask
            mask_path = os.path.join(mask_dir, f"{base_name}_manual1.gif")
            with Image.open(mask_path) as source:
                mask = (np.asarray(source.convert('L')) > 127).astype(np.float32)[..., None]
            if mask.shape[:2] != img.shape[:2] or min(img.shape[:2]) < self.patch_size:
                raise ValueError(f'Invalid image/mask dimensions for {filename}')
            self.images.append(img)
            self.masks.append(mask)
        if not self.images:
            raise ValueError('No paired DRIVE training images and masks found.')

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        batch_x = np.zeros((self.batch_size, self.patch_size, self.patch_size, 3), dtype=np.float32)
        batch_y = np.zeros((self.batch_size, self.patch_size, self.patch_size, 1), dtype=np.float32)
        
        for i in range(self.batch_size):
            # Select random image
            img_idx = np.random.randint(0, len(self.images))
            img = self.images[img_idx]
            mask = self.masks[img_idx]
            
            # Slice random patch
            h, w = img.shape[:2]
            y = np.random.randint(0, h - self.patch_size + 1)
            x = np.random.randint(0, w - self.patch_size + 1)
            
            img_patch = img[y:y+self.patch_size, x:x+self.patch_size]
            mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
            
            # Apply geometric augmentation
            if np.random.rand() > 0.5:
                img_patch = np.fliplr(img_patch)
                mask_patch = np.fliplr(mask_patch)
            if np.random.rand() > 0.5:
                img_patch = np.flipud(img_patch)
                mask_patch = np.flipud(mask_patch)
                
            batch_x[i] = img_patch
            batch_y[i] = mask_patch
            
        return batch_x, batch_y

# --- 4. EXECUTE TRAINING ---
if __name__ == '__main__':
    img_dir = os.path.join(DRIVE_TRAIN_DIR, 'images')
    mask_dir = os.path.join(DRIVE_TRAIN_DIR, '1st_manual')
    
    print("Building memory-safe data generator...")
    train_gen = DRIVEDatasetGenerator(img_dir, mask_dir, batch_size=BATCH_SIZE, patch_size=PATCH_SIZE, steps_per_epoch=STEPS_PER_EPOCH)
    
    print("Building and compiling U-Net...")
    model = build_unet()
    
    callbacks = [
        ModelCheckpoint('unet_vessel_segmentation.keras', save_best_only=True, monitor='loss'),
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    ]
    
    print("Initiating Training...")
    model.fit(
        train_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    print("\n✅ U-Net Vessel Segmentation Model saved as 'unet_vessel_segmentation.keras'")
