import gdown
import os
print("Import success")

def download_google_drive_models():
    path = os.getcwd()
    if os.path.exists(path+"/img_model.pkl") == False:
        img_model_id = "1vUdwHPWhIYTb1xhdxtdyo0Erl4PEsIQe"
        gdown.download(id = img_model_id, output=path+"/img_model.pkl", fuzzy=True)
        print("Downloaded img model")
    if os.path.exists(path+"/txt_model.pkl") == False:
        txt_model_id = "1rLdZBufn_d0UHZfAB51xw1R41SKgEffB"
        gdown.download(id = txt_model_id, output=path+"/txt_model.pkl", fuzzy=True)
        print("Downloaded txt model")