import os                                           # FOR WORKING WITH FILES IN THIS COMPUTER
import shutil                                       # FOR MOVING COPYING STUFFS 
from pathlib import Path                            # HUNDLING FILE/FOLDERS PATH
from tqdm import tqdm                               # SHOWING A NICE BAR ANIMATION WHEN WE READING IMAGES 
from PIL import Image                               # PILLOW LABERARIE - READ AND OPEN IMAGES FILES 
import torch                                        # AI FRAMEWORK WE'RE USING
import torchvision.transforms as transforms         # TOOLS TO PREPARE IMAGE FOR THE AI-MODEL
import torchvision.models as models                 # WE'LL USE RESNET50 - READY-MADE AI MODEL
import json                                         # READING THE LIST OF CLASS NAMES 
import urllib.request                               # DOWNLOADING THE CLASS NAMES IF NEEDED

# =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*= =* =*        USE IT IN HALLAL PLEASE      =* =* =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=

# ==================== STEP 1: Load the AI Model =======================================
print("LOADING THE RESNET50 MODEL , THIS MAY TAKE CHWYA WA9T PLEASE OSBOR CHWYA... ")
Model_000 = models.resnet50(pretrained=True)                                           
Model_000.eval()                                                                       
# ==================== STEP 2: Get the Names of Categories ====================
Labels_path = "imagenet_classes.txt"                                                   
if not os.path.exists(Labels_path):                                                     
    print("Downloading human-readable ImageNet class names...")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json",
                                Labels_path)
with open(Labels_path, "r" , encoding="utf-8") as f:
    imagenet_classes = json.load(f)                                                     
# ==================== STEP 3: Prepare Images for the Model ============================
transform = transforms.Compose ([                                                      
            transforms.Resize(456),                                                     
            transforms.CenterCrop(456),                                                  
            transforms.ToTensor(),                                                      
            transforms.Normalize(                                                        
                mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
# ==================== STEP 4: Function to Predict What’s in an Image ==================
def predict_image_catergory(image_path):                                                
    try:
        img = Image.open(image_path).convert("RGB")                                     
        img_t = transform(img)                                                          
        batch_t = torch.unsqueeze(img_t, 0)                                             
        with torch.no_grad():                                                           
            output = Model_000(batch_t)                                                 
        _, index = torch.max(output, 1)                                                 
        pred_class_name = imagenet_classes[index.item()]                                

        percentages = torch.nn.functional.softmax(output[0], dim=0)                     
        top3_idx = percentages.topk(10).indices                                         
        top3_names = [imagenet_classes[i] for i in top3_idx]                            
        return pred_class_name, top3_names
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return "error", []
# ==================== STEP 5: Decide Which Folder Each Image Should Go To ============
category_mapping = {
  "person": ["person", "man", "woman", "child", "people", "bride", "groom", "baby", "teenager", "elderly", "group of people", "portrait", "selfie", "family", "couple", "athlete", "model", "businessman", "doctor", "police officer"],
  "car": ["car", "sports car", "convertible", "jeep", "truck", "limousine", "race car", "suv", "sedan", "hatchback", "pickup truck", "van", "bus", "motorcycle"],
  "airplane": ["airplane", "jet", "helicopter", "propeller plane", "fighter jet", "drone", "hot air balloon"],
  "cartoon": ["comic book", "cartoon", "mask", "anime", "superhero", "disney character", "animated", "manga", "chibi", "pixar style"],
  "phone": ["phone", "smartphone", "iphone", "android phone", "flip phone", "rotary phone", "cell phone", "mobile phone", "tablet", "smartwatch"],
  "bicycle": ["bicycle", "bike", "mountain bike", "road bike", "bmx", "electric bike", "scooter"],
  "boat": ["boat", "sailboat", "yacht", "ship", "canoe", "kayak", "ferry", "speedboat", "submarine", "cruise ship"],
  "furniture": ["chair", "sofa", "couch", "table", "bed", "dining table", "bench", "desk", "bookshelf", "wardrobe", "cabinet", "armchair", "coffee table", "tv stand", "lamp"],
  "food": ["apple", "banana", "orange", "pizza", "sandwich", "cake", "donut", "burger", "salad", "pasta", "sushi", "ice cream", "coffee", "wine", "bread", "cheese", "steak", "soup", "breakfast"],
  "animal": ["horse","butterfly", "bee", "ladybug", "ant", "spider", "dragonfly", "beetle", "zebra", "pony", "stallion", "foal", "unicorn","bird", "parrot", "eagle", "owl", "penguin", "flamingo", "sparrow", "hawk", "duck", "swan", "peacock", "hummingbird", "robin","cat", "kitten", "persian", "siamese", "tabby", "maine coon", "sphynx", "british shorthair", "ragdoll","dog", "puppy", "retriever", "terrier", "beagle", "husky", "poodle", "bulldog", "labrador", "german shepherd", "dalmatian", "chihuahua", "pug", "corgi","sheep", "cow", "elephant", "giraffe", "bear", "lion", "tiger", "monkey", "wolf", "fox", "deer", "rabbit", "squirrel", "panda", "koala", "kangaroo", "snake", "crocodile"],
  "tree": ["flower", "rose", "tulip", "sunflower", "daisy", "lily", "orchid", "lavender", "cherry blossom", "lotus","tree", "pine tree", "palm tree", "oak tree", "maple", "christmas tree", "birch", "willow", "bamboo"],
  "building": ["building", "house", "skyscraper", "castle", "church", "office building", "apartment", "villa", "cottage", "bridge", "tower", "factory"],
  "landscape": ["mountain", "beach", "forest", "desert", "waterfall", "lake", "river", "ocean", "sunset", "snow landscape", "meadow", "valley", "canyon", "island", "volcano"],
  "clothing": ["dress", "shirt", "pants", "jacket", "shoes", "hat", "tie", "skirt", "sweater", "jeans", "t-shirt", "hoodie", "coat", "boots", "sneakers", "gloves", "scarf"],
  "electronics": ["laptop", "computer", "monitor", "keyboard", "mouse", "tv", "camera", "headphones", "speaker", "printer", "game console", "router", "earbuds"],
  "sports": ["soccer ball", "basketball", "tennis racket", "football", "golf club", "baseball bat", "swimming", "running", "skiing", "cycling", "boxing", "yoga", "gym", "surfing"],
  "musical_instrument": ["guitar", "piano", "drum", "violin", "trumpet", "saxophone", "microphone", "flute", "cello", "bass guitar", "keyboard synthesizer", "harp"],
  "book": ["book", "novel", "magazine", "newspaper", "comic book", "textbook", "notebook", "library"],
  "fruit": ["apple", "banana", "orange", "grape", "strawberry", "watermelon", "pineapple","carrot", "potato", "tomato", "broccoli", "onion", "lettuce", "cucumber", "pepper", "corn", "garlic", "spinach", "mango", "pear", "cherry", "lemon", "kiwi"],
  "weather": ["sunny", "rain", "snow", "cloudy", "storm", "fog", "rainbow", "lightning", "windy"],
  "kitchenware": ["plate", "cup", "glass", "fork", "knife", "spoon", "bowl", "pot", "pan", "oven", "fridge", "microwave"],
  "toy": ["teddy bear", "doll", "lego", "toy car", "puzzle", "ball", "action figure", "board game","weapon"],
  "art_style": ["painting", "oil painting", "watercolor", "digital art", "sketch", "abstract art", "realistic", "surreal", "graffiti"],
  "interior_design": ["living room", "bedroom", "kitchen", "bathroom", "minimalist interior", "scandinavian style", "modern home", "luxury interior", "vintage decor", "industrial style", "bohemian", "rustic"],
  "product_setup": ["studio lighting", "white background", "flat lay", "lifestyle shot", "macro closeup", "high key photography", "commercial product shot", "packshot", "shadow play", "floating product", "ecommerce photo", "hero shot"]
}
def get_folder_name(pred_class, top_preds):
    scores = {folder: 0 for folder in category_mapping}
    all_preds = [pred_class.lower()] + [p.lower() for p in top_preds]
    for folder, keywords in category_mapping.items():
        for kw in keywords:
            for pred in all_preds:
                if kw in pred:
                    scores[folder] += 1
    best_folder = max(scores, key=scores.get)
    if scores[best_folder] == 0:
        return "others"
    return best_folder
# ==================== STEP 6: Main Function - Organize All Images ====================
def organize_images(main_folder_path, copy_instead_of_move=False):
    main_folder = Path(main_folder_path)
    if not main_folder.exists():
        print(f"Error: Folder not found → {main_folder_path}")
        return
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp", ".JPG", ".JPEG", ".PNG"}
    image_files = [
        f for f in main_folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    if not image_files:
        print("No images found in this folder.")
        return
    print(f"Found {len(image_files)} images. Starting classification and organizing...\n")
    for image_path in tqdm(image_files):
        main_class, top3 = predict_image_catergory(image_path)       
        if main_class == "error":
            continue 
        target_folder_name = get_folder_name(main_class, top3)
        target_folder = main_folder / target_folder_name
        target_folder.mkdir(exist_ok=True) 
        destination = target_folder / image_path.name
        counter = 1
        while destination.exists():
            destination = target_folder / f"{image_path.stem}_{counter}{image_path.suffix}"
            counter += 1
        if copy_instead_of_move:
            shutil.copy2(image_path, destination)
            print(f"Copied → {target_folder_name}/{destination.name}")
        else:
            shutil.move(str(image_path), str(destination))
            print(f"Moved → {target_folder_name}/{destination.name}")
# ==================== STEP 7: RUN THE SCRIPT ===========================================
FOLDER_TO_ORGANIZE = r"             " #____________<-----------________file you want to organize_______________
organize_images(FOLDER_TO_ORGANIZE, copy_instead_of_move=True)
print("\nAll done! Check your folders.")
