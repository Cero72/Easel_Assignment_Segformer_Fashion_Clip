import matplotlib.pyplot as plt

# Define the class names for the segmentation model
class_names = [
    "Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt", "Pants", 
    "Dress", "Belt", "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", 
    "Left-arm", "Right-arm", "Bag", "Scarf"
]

# Define a color map for visualization
color_map = plt.cm.get_cmap('tab20', len(class_names))

# Define a mapping of garment types to related segments that should be included
garment_to_segments = {
    0: [0],                   # Background --> segment background only
    1: [1, 2, 11],            # Hat --> segment hat, hair, and face
    2: [2],                   # Hair --> segment hair only
    3: [3, 11],               # Sunglasses --> segment sunglasses and face
    4: [4, 14, 15],           # Upper-clothes --> segment upper clothes, left arm, right arm
    5: [5, 6, 12, 13],        # Skirt --> segment skirt, pants, left leg, right leg
    6: [6, 12, 13],           # Pants --> segment pants, left leg, right leg
    7: [4, 5, 6, 7, 12, 13, 14, 15],  # Dress --> segment whole body except face and hair
    8: [8],                   # Belt --> segment belt only
    9: [9],                   # Left-shoe --> segment left shoe only
    10: [10],                 # Right-shoe --> segment right shoe only
    11: [11],                 # Face --> segment face only
    12: [12],                 # Left-leg --> segment left leg only
    13: [13],                 # Right-leg --> segment right leg only
    14: [14],                 # Left-arm --> segment left arm only
    15: [15],                 # Right-arm --> segment right arm only
    16: [16],                 # Bag --> segment bag only
    17: [17, 2, 11]           # Scarf --> segment scarf, hair and face
}

# Define categories for Fashion-CLIP
fashion_categories = [
    # Upper body
    "t-shirt", "shirt", "blouse", "tank top", "polo shirt", "sweatshirt", "hoodie",
    
    # Outerwear
    "jacket", "coat", "blazer", "cardigan", "vest", "windbreaker",
    
    # Dresses
    "dress", "shirt dress", "sundress", "evening gown", "maxi dress", "mini dress",
    
    # Lower body
    "jeans", "pants", "trousers", "shorts", "skirt", "leggings", "joggers", "sweatpants",
    
    # Formal wear
    "suit", "tuxedo", "formal shirt", "formal dress",
    
    # Undergarments
    "bra", "underwear", "boxers", "briefs", "lingerie",
    
    # Sleepwear
    "pajamas", "nightgown", "bathrobe",
    
    # Swimwear
    "swimsuit", "bikini", "swim trunks",
    
    # Footwear
    "shoes", "boots", "sneakers", "sandals", "high heels", "loafers", "flats",
    
    # Accessories
    "hat", "cap", "scarf", "gloves", "belt", "tie", "socks",
    
    # Bags
    "handbag", "backpack", "purse", "tote bag"
]

# Mapping from Fashion-CLIP categories to SegFormer classes
fashion_clip_to_segformer = {
    # Upper body items -> Upper-clothes (4)
    "t-shirt": 4, "shirt": 4, "blouse": 4, "tank top": 4, "polo shirt": 4, "sweatshirt": 4, "hoodie": 4,
    "cardigan": 4, "vest": 4, "formal shirt": 4,
    
    # Outerwear -> Upper-clothes (4)
    "jacket": 4, "coat": 4, "blazer": 4, "windbreaker": 4,
    
    # Dresses -> Dress (7)
    "dress": 7, "shirt dress": 7, "sundress": 7, "evening gown": 7, "maxi dress": 7, "mini dress": 7,
    "formal dress": 7,
    
    # Lower body -> Pants (6) or Skirt (5)
    "jeans": 6, "pants": 6, "trousers": 6, "shorts": 6, "leggings": 6, "joggers": 6, "sweatpants": 6,
    "skirt": 5,
    
    # Formal wear -> Upper-clothes (4) or Dress (7)
    "suit": 4, "tuxedo": 4,
    
    # Footwear -> Left-shoe/Right-shoe (9/10)
    "shoes": 9, "boots": 9, "sneakers": 9, "sandals": 9, "high heels": 9, "loafers": 9, "flats": 9,
    
    # Accessories
    "hat": 1, "cap": 1, "scarf": 17, "belt": 8,
    
    # Bags
    "handbag": 16, "backpack": 16, "purse": 16, "tote bag": 16
}

# Detailed mapping from categories to segment names
category_to_segment_mapping = {
    # Upper body items map to Upper-clothes and arms
    "t-shirt": ["Upper-clothes", "Left-arm", "Right-arm"],
    "shirt": ["Upper-clothes", "Left-arm", "Right-arm"],
    "blouse": ["Upper-clothes", "Left-arm", "Right-arm"],
    "tank top": ["Upper-clothes", "Left-arm", "Right-arm"],
    "polo shirt": ["Upper-clothes", "Left-arm", "Right-arm"],
    "sweatshirt": ["Upper-clothes", "Left-arm", "Right-arm"],
    "hoodie": ["Upper-clothes", "Left-arm", "Right-arm"],
    
    # Outerwear maps to Upper-clothes and arms
    "jacket": ["Upper-clothes", "Left-arm", "Right-arm"],
    "coat": ["Upper-clothes", "Left-arm", "Right-arm"],
    "blazer": ["Upper-clothes", "Left-arm", "Right-arm"],
    "cardigan": ["Upper-clothes", "Left-arm", "Right-arm"],
    "vest": ["Upper-clothes"],
    "windbreaker": ["Upper-clothes", "Left-arm", "Right-arm"],
    
    # Dresses map to Dress
    "dress": ["Dress", "Upper-clothes", "Skirt", "Pants", "Left-arm", "Right-arm", "Left-leg", "Right-leg"],
    "shirt dress": ["Dress", "Upper-clothes", "Skirt", "Pants", "Left-arm", "Right-arm", "Left-leg", "Right-leg"],
    "sundress": ["Dress", "Upper-clothes", "Skirt", "Pants", "Left-arm", "Right-arm", "Left-leg", "Right-leg"],
    "evening gown": ["Dress", "Upper-clothes", "Skirt", "Pants", "Left-arm", "Right-arm", "Left-leg", "Right-leg"],
    "maxi dress": ["Dress", "Upper-clothes", "Skirt", "Pants", "Left-arm", "Right-arm", "Left-leg", "Right-leg"],
    "mini dress": ["Dress", "Upper-clothes", "Skirt", "Pants", "Left-arm", "Right-arm", "Left-leg", "Right-leg"],
    "formal dress": ["Dress", "Upper-clothes", "Skirt", "Pants", "Left-arm", "Right-arm", "Left-leg", "Right-leg"],
    
    # Lower body items map to Pants or Skirt and legs
    "jeans": ["Pants", "Left-leg", "Right-leg"],
    "pants": ["Pants", "Left-leg", "Right-leg"],
    "trousers": ["Pants", "Left-leg", "Right-leg"],
    "shorts": ["Pants", "Left-leg", "Right-leg"],
    "skirt": ["Skirt", "Pants", "Left-leg", "Right-leg"],
    "leggings": ["Pants", "Left-leg", "Right-leg"],
    "joggers": ["Pants", "Left-leg", "Right-leg"],
    "sweatpants": ["Pants", "Left-leg", "Right-leg"],
    
    # Formal wear maps depending on type
    "suit": ["Upper-clothes", "Left-arm", "Right-arm", "Pants", "Left-leg", "Right-leg"],
    "tuxedo": ["Upper-clothes", "Left-arm", "Right-arm", "Pants", "Left-leg", "Right-leg"],
    "formal shirt": ["Upper-clothes", "Left-arm", "Right-arm"],
    "formal dress": ["Dress", "Upper-clothes", "Skirt", "Pants", "Left-arm", "Right-arm", "Left-leg", "Right-leg"],
    
    # Footwear maps to shoes
    "shoes": ["Left-shoe", "Right-shoe"],
    "boots": ["Left-shoe", "Right-shoe"],
    "sneakers": ["Left-shoe", "Right-shoe"],
    "sandals": ["Left-shoe", "Right-shoe"],
    "high heels": ["Left-shoe", "Right-shoe"],
    "loafers": ["Left-shoe", "Right-shoe"],
    "flats": ["Left-shoe", "Right-shoe"],
    
    # Accessories map to their respective parts
    "hat": ["Hat"],
    "cap": ["Hat"],
    "scarf": ["Scarf", "Face", "Hair"],
    "gloves": ["Left-arm", "Right-arm"],
    "belt": ["Belt"],
    "tie": ["Upper-clothes"],
    "socks": ["Left-leg", "Right-leg"],
    
    # Bags map to Bag
    "handbag": ["Bag"],
    "backpack": ["Bag"],
    "purse": ["Bag"],
    "tote bag": ["Bag"]
}
