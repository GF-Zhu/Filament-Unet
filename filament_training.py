from filament_model import *
from data import *
import pydotplus
import time
start =time.clock()

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

data_gen_args = dict(rotation_range=0.2,#0.2
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')


myGene = trainGenerator(2,'data/filament/train','image','label',data_gen_args,save_to_dir = 'data/filament/train/aug/')
model = filament_unet()

model_checkpoint = ModelCheckpoint('filament_unet_cpu.hdf5', monitor='loss',verbose=1, save_best_only=True,save_weights_only=True)
model_EarlyStopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=3, verbose=0, mode='auto')
history = model.fit_generator(myGene,steps_per_epoch=300,epochs=20,shuffle=True,callbacks=[model_checkpoint,model_EarlyStopping])

end=time.clock()
print('Running time: %s Seconds'%(end-start))




