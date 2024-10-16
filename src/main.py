
import images
import img_processing

def driver():
    chest_train_img_processor = img_processing.VisibilityDetection('data/images/Chest-RSNA/train/good', a0=500, a1=0.1)

driver()