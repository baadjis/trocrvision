import sys
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from tqdm.auto import tqdm
from urllib.request import urlretrieve
from zipfile import ZipFile
 
 
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import glob

#device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
#define the device
device = torch.device('cpu')

def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assets....", end="")
 
 
    # Downloading zip file using urllib package.
    urlretrieve(url, save_path)
 
 
    try:
        # Extracting zip file using the zipfile package.
        with ZipFile(save_path) as z:
            # Extract ZIP file contents in the same directory.
            z.extractall(os.path.split(save_path)[0])
 
 
        print("Done")
 
 
    except Exception as e:
        print("\nInvalid file.", e)
 
URL = r"https://www.dropbox.com/scl/fi/jz74me0vc118akmv5nuzy/images.zip?rlkey=54flzvhh9xxh45czb1c8n3fp3&dl=1"
asset_zip_path = os.path.join(os.getcwd(), "images.zip")
# Download if assest ZIP does not exists.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)





def read_image(image_path):
    """
    read image from path and convert to RGB
    :param image_path: String, path to the input image.
 
 
    Returns:
        image: PIL Image.
    """
    image = Image.open(image_path).convert('RGB')
    return image




def compute_ocr(image, processor, model):
    """
    :param image: PIL Image.
    :param processor: Huggingface OCR processor.
    :param model: Huggingface OCR model.
 
 
    Returns:
        generated_text: the OCR'd text string.
    """
    # We can directly perform OCR on cropped images.
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def plot_image_text(image,text:str):
    """plot input image and write predicted text

    Args:
        image (Image): input image
        text (str): output text
    """
    plt.figure(figsize=(7, 4))
    plt.imshow(image)
    plt.title(text)
    plt.axis('off')
    plt.show()


def predict(processor,data_path=None, num_samples=4, model=None):
    image_paths = glob.glob(data_path)
    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        if i == num_samples:
            break
        image = read_image(image_path)
        text = compute_ocr(image, processor, model)
        plot_image_text(image=image,text=text)
        
        
def predict_newspaper():

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
    model = VisionEncoderDecoderModel.from_pretrained(
        'microsoft/trocr-small-printed'
    ).to(device)

    predict(processor=processor,
        data_path=os.path.join('images', 'newspaper', '*'),
        num_samples=2,
        model=model
    )

def predict_handwritten():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained(
        'microsoft/trocr-base-handwritten'
    ).to(device)


    predict(processor=processor,
        data_path=os.path.join('images', 'handwritten', '*'),
        num_samples=2,
        model=model
    )

def main():
    if len(sys.argv)==1:
        print("add argments: handwritten or newspaper")
    else:
        arg=sys.argv[1]
        if arg=="handwritten":
            predict_handwritten()
        elif arg=="newspaper":
            predict_newspaper()
        else:
            print("invalid argment use handwritten or newspaper")
            

if __name__=='__main__':
    main()