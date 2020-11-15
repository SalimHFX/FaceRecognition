import h5py

# H5 File structure :
# Option 1 : X_train = [face0.pmg,face1.pmg], Y_train = [noface0.pmg, noface1.pmg]
# Option 2 : X_train = [face0.pmg,face1.pmg,noface0.pmg,noface1.pmg], Y_train = ['face','face','noface','noface']

# Classic torch ImageFolder => Expects each class to be in a subfolder

#Create the h5 file
fileName = 'train_set.h5'
numOfSamples = 4
with h5py.File(fileName, "w") as out:
  out.create_dataset("X_train",(numOfSamples,36,36),dtype='u1')
  out.create_dataset("Y_train",(numOfSamples,1,1),dtype='u1')


