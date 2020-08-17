import data_gen as imdata

# imdata.store_dataset("imdata")
imdata.load_dataset("imdata")

(img_train, label_train), (img_validation, label_validation), (img_test, label_test) = imdata.load_dataset(\
    "/Users/do-hyungkwon/GoogleDrive/jupyter_notebook/imdata")

print(img_train.shape)
print(label_train.shape)
print(img_validation.shape)
print(label_validation.shape)
print(img_test.shape)
print(label_test.shape)