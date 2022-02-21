from typing import Tuple

import sys
import torch
import random
import hashlib

_INTERACTIVE_OBJECTS = [
    'AlarmClock',
    'Apple',
    'ArmChair',
    'BaseballBat',
    'BasketBall',
    'Bathtub',
    'BathtubBasin',
    'Bed',
    'Blinds',
    'Book',
    'Boots',
    'Bowl',
    'Box',
    'Bread',
    'ButterKnife',
    'Cabinet',
    'Candle',
    'Cart',
    'CD',
    'CellPhone',
    'Chair',
    'Cloth',
    'CoffeeMachine',
    'CounterTop',
    'CreditCard',
    'Cup',
    'Curtains',
    'Desk',
    'DeskLamp',
    'DishSponge',
    'Drawer',
    'Dresser',
    'Egg',
    'FloorLamp',
    'Footstool',
    'Fork',
    'Fridge',
    'GarbageCan',
    'Glassbottle',
    'HandTowel',
    'HandTowelHolder',
    'HousePlant',
    'Kettle',
    'KeyChain',
    'Knife',
    'Ladle',
    'Laptop',
    'LaundryHamper',
    'LaundryHamperLid',
    'Lettuce',
    'LightSwitch',
    'Microwave',
    'Mirror',
    'Mug',
    'Newspaper',
    'Ottoman',
    'Painting',
    'Pan',
    'PaperTowel',
    'PaperTowelRoll',
    'Pen',
    'Pencil',
    'PepperShaker',
    'Pillow',
    'Plate',
    'Plunger',
    'Poster',
    'Pot',
    'Potato',
    'RemoteControl',
    'Safe',
    'SaltShaker',
    'ScrubBrush',
    'Shelf',
    'ShowerDoor',
    'ShowerGlass',
    'Sink',
    'SinkBasin',
    'SoapBar',
    'SoapBottle',
    'Sofa',
    'Spatula',
    'Spoon',
    'SprayBottle',
    'Statue',
    'StoveBurner',
    'StoveKnob',
    'DiningTable',
    'CoffeeTable',
    'SideTable',
    'TeddyBear',
    'Television',
    'TennisRacket',
    'TissueBox',
    'Toaster',
    'Toilet',
    'ToiletPaper',
    'ToiletPaperHanger',
    'ToiletPaperRoll',
    'Tomato',
    'Towel',
    'TowelHolder',
    'TVStand',
    'Vase',
    'Watch',
    'WateringCan',
    'Window',
    'WineBottle',
]

_TABLETOP_OBJECTS = [
    'AlarmClock',
    'Apple',
    'BaseballBat',
    'BasketBall',
    'Book',
    'Boots',
    'Bowl',
    'Box',
    'Bread',
    'ButterKnife',
    'Cabinet',
    'Candle',
    'Cart',
    'CD',
    'CellPhone',
    'Cloth',
    'CoffeeMachine',
    'CreditCard',
    'Cup',
    'DeskLamp',
    'DishSponge',
    'Egg',
    'FloorLamp',
    'Footstool',
    'Fork',
    'GarbageCan',
    'Glassbottle',
    'HandTowel',
    'HandTowelHolder',
    'Kettle',
    'KeyChain',
    'Knife',
    'Ladle',
    'Laptop',
    'Lettuce',
    'Microwave',
    'Mug',
    'Newspaper',
    'Pan',
    'PaperTowel',
    'PaperTowelRoll',
    'Pen',
    'Pencil',
    'PepperShaker',
    'Pillow',
    'Plate',
    'Plunger',
    'Pot',
    'Potato',
    'RemoteControl',
    'Safe',
    'SaltShaker',
    'ScrubBrush',
    'Sink',
    'SinkBasin',
    'SoapBar',
    'SoapBottle',
    'Spatula',
    'Spoon',
    'SprayBottle',
    'Statue',
    'TeddyBear',
    'TennisRacket',
    'TissueBox',
    'Toaster',
    'ToiletPaper',
    'ToiletPaperHanger',
    'ToiletPaperRoll',
    'Tomato',
    'Towel',
    'TowelHolder',
    'Vase',
    'Watch',
    'WateringCan',
    'WineBottle',
]


_STRUCTURAL_OBJECTS = [
    "Books",
    "Ceiling",
    "Door",
    "Floor",
    "KitchenIsland",
    "LightFixture",
    "Rug",
    "Wall",
    "StandardWallSize",
    "Faucet",
    "Bottle",
    "Bag",
    "Cube",
    "Room",
]

# Objects that tend to come in different colors and textures
AUGMENTATION_OBJECTS = _STRUCTURAL_OBJECTS + [
    'ArmChair',
    'Bathtub',
    'BathtubBasin',
    'Bed',
    'Blinds',
    'Cabinet',
    'Chair',
    'Cloth',
    'CoffeeMachine',
    'CounterTop',
    'Curtains',
    'Desk',
    'DeskLamp',
    'Drawer',
    'Dresser',
    'FloorLamp',
    'Footstool',
    'Fridge',
    'GarbageCan',
    'HandTowel',
    'HandTowelHolder',
    'LaundryHamper',
    'LaundryHamperLid',
    'LightSwitch',
    'Microwave',
    'Ottoman',
    'Painting',
    'Pillow',
    'Poster',
    'Shelf',
    'ShowerDoor',
    'ShowerGlass',
    'Sink',
    'SinkBasin',
    'Sofa',
    'SoapBottle',
    'SprayBottle',
    'Statue',
    'DiningTable',
    'CoffeeTable',
    'SideTable',
    'ToiletPaperHanger',
    'Towel',
    'TVStand',
    'Vase',
    'WateringCan',
    'Window',
]

OBJECT_STR_TO_DESCR = {
    'AlarmClock' : "Alarm clock",
    'BaseballBat' : "Baseball bat",
    'BathtubBasin': "Bathtub basin",
    'ButterKnife' : "Butter knife",
    'CoffeeMachine' : "Coffee machine",
    'CreditCard' : "Credit card",
    'DeskLamp' : "Desk lamp",
    'DishSponge': "Dish sponge",
    'FloorLamp' : "Floor lamp",
    'GarbageCan' : "Garbage can",
    'Glassbottle' : "Glass bottle",
    'HandTowel' : "Hand towel",
    'HandTowelHolder' : "Towel holder",
    'HousePlant' : "Plant",
    'LaundryHamper' : "Laundry hamper",
    'LaundryHamperLid' : "Laundry hamper lid",
    'LightSwitch' : "Light switch",
    'PaperTowel' : "Paper towel",
    'PaperTowelRoll' : "Paper towel roll",
    'PepperShaker' : "Pepper shaker",
    'RemoteControl' : "Remote control",
    'SaltShaker' : "Salt shaker",
    'ScrubBrush' : "Scrub brush",
    'ShowerDoor' : "Shower door",
    'ShowerGlass' : "Shower glass",
    'SinkBasin' : "Sink",
    'SoapBar' : "Soap bar",
    'SoapBottle' : "Soap bottle",
    'SprayBottle' : "Spray bottle",
    'StoveBurner' : "Stove burner",
    'StoveKnob' : "Stove knob",
    'DiningTable' : "Dining table",
    'CoffeeTable' : "Coffee table",
    'SideTable' : "Side table",
    'TennisRacket' : "Tennis racket",
    'TissueBox' : "Tissue box",
    'ToiletPaper' : "Toilet paper",
    'ToiletPaperHanger' : "Toilet paper holder",
    'ToiletPaperRoll' : "Toilet paper roll",
    'TowelHolder' : "Towel holder",
    'TVStand' : "TV Stand",
    'WateringCan' : "Watering can",
    'WineBottle' : "Wine bottle",
}

INVENTORY_OBJECT_STR = "<InventoryObject>"
_EXTRA_TOKENS = [
    INVENTORY_OBJECT_STR
]

_RECEPTACLE_OBJECTS = [
    'BathtubBasin',
    'Bowl',
    'Cup',
    'Drawer',
    'Mug',
    'Plate',
    'Shelf',
    'SinkBasin',
    'Box',
    'Cabinet',
    'CoffeeMachine',
    'CounterTop',
    'Fridge',
    'GarbageCan',
    'HandTowelHolder',
    'Microwave',
    'PaintingHanger',
    'Pan',
    'Pot',
    'StoveBurner',
    'DiningTable',
    'CoffeeTable',
    'SideTable',
    'ToiletPaperHanger',
    'TowelHolder',
    'Safe',
    'BathtubBasin',
    'ArmChair',
    'Toilet',
    'Sofa',
    'Ottoman',
    'Dresser',
    'LaundryHamper',
    'Desk',
    'Bed',
    'Cart',
    'TVStand',
    'Toaster',
]

_MOVABLE_RECEPTACLES = [
    'Bowl',
    'Box',
    'Cup',
    'Mug',
    'Plate',
    'Pan',
    'Pot',
]

_OPENABLES = ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'Safe', 'Box']

_TOGGLABLES = ["DeskLamp", "FloorLamp", "Microwave"]

_PICKABLES = [s for s in _INTERACTIVE_OBJECTS if (
        (s not in _RECEPTACLE_OBJECTS) and
        (s not in _OPENABLES) and
        (s not in _TOGGLABLES)) or (s in _MOVABLE_RECEPTACLES)
    ]


OBJECT_CLASSES = _STRUCTURAL_OBJECTS + _INTERACTIVE_OBJECTS + _EXTRA_TOKENS

# Mappings between integers and strings
OBJECT_INT_TO_STR = {i: o for i, o in enumerate(OBJECT_CLASSES)}
OBJECT_STR_TO_INT = {o: i for i, o in enumerate(OBJECT_CLASSES)}
UNK_OBJ_INT = len(OBJECT_CLASSES)
UNK_OBJ_STR = "Unknown"
#COLOR_OTHERS = (255, 0, 0)
COLOR_OTHERS = (100, 100, 100)


# Mappings between integers and colors
def _compute_object_intid_to_color_o(object_intid: int) -> Tuple[int, int, int]:
    # Backup and restore random number generator state so as not to mess with the rest of the project
    # (e.g. if a random seed is fixed, we want to keep it fixed. If it isn't we don't want to fix it.)
    randstate = random.getstate()
    random.seed(object_intid)
    color = tuple(random.randint(50, 240) for _ in range(3))
    random.setstate(randstate)
    return color

def _compute_object_intid_to_color(object_intid: int) -> Tuple[int, int, int]:
    # Backup and restore random number generator state so as not to mess with the rest of the project
    # (e.g. if a random seed is fixed, we want to keep it fixed. If it isn't we don't want to fix it.)
    int_hash = int.from_bytes(hashlib.md5(str(object_intid).encode()).digest(), byteorder=sys.byteorder)
    r = 20 + int_hash % 200
    g = 20 + int(int_hash / 200) % 200
    b = 20 + int(int_hash / (200 * 2)) % 200
    color = (r, g, b)
    return color

# Precompute color for each object
OBJECT_INTID_TO_COLOR = {i: _compute_object_intid_to_color(i) for i in range(len(OBJECT_CLASSES))}
OBJECT_INTID_TO_COLOR[UNK_OBJ_INT] = COLOR_OTHERS
OBJECT_COLOR_TO_INTID = {c: i for i, c in OBJECT_INTID_TO_COLOR.items()}


# -------------------------------------------------------------
# Public API:
# -------------------------------------------------------------
# Simple mappings

def get_all_interactive_objects():
    return list(iter(_INTERACTIVE_OBJECTS))

def get_receptacle_ids():
    return [object_string_to_intid(s) for s in _RECEPTACLE_OBJECTS]

def get_pickable_ids():
    return [object_string_to_intid(s) for s in _PICKABLES]

def get_togglable_ids():
    return [object_string_to_intid(s) for s in _TOGGLABLES]

def get_openable_ids():
    return [object_string_to_intid(s) for s in _OPENABLES]

def get_ground_ids():
    return [object_string_to_intid(s) for s in ["Rug", "Floor"]]

def get_num_objects():
    return len(OBJECT_CLASSES) + 1

def object_color_to_intid(color: Tuple[int, int, int]) -> int:
    global OBJECT_COLOR_TO_INTID
    return OBJECT_COLOR_TO_INTID[color]

def object_intid_to_color(intid: int) -> Tuple[int, int, int]:
    global OBJECT_INTID_TO_COLOR
    return OBJECT_INTID_TO_COLOR[intid]

def object_string_to_intid(object_str) -> int:
    global OBJECT_STR_TO_INT, UNK_OBJ_INT
    # Remove the part about object instance location
    object_str = object_str.split("|")[0].split(":")[-1].split(".")[0]
    if object_str in OBJECT_STR_TO_INT:
        return OBJECT_STR_TO_INT[object_str]
    else:
        return UNK_OBJ_INT

INTERACTIVE_OBJECT_IDS = [object_string_to_intid(o) for o in _INTERACTIVE_OBJECTS]
STRUCTURAL_OBJECT_IDS = [object_string_to_intid(o) for o in _STRUCTURAL_OBJECTS]
TABLETOP_OBJECT_IDS = [object_string_to_intid(o) for o in _TABLETOP_OBJECTS]

def object_intid_to_string(intid: int) -> str:
    global OBJECT_INT_TO_STR, UNK_OBJ_STR
    if intid in OBJECT_INT_TO_STR:
        return OBJECT_INT_TO_STR[intid]
    else:
        return UNK_OBJ_STR

def object_string_to_color(object_str : str) -> Tuple[int, int, int]:
    return object_intid_to_color(object_string_to_intid((object_str)))

def object_color_to_string(color: Tuple[int, int, int]) -> str:
    return object_intid_to_string(object_color_to_intid(color))


# Image-related definitions

def get_class_color_vector():
    colors = [object_intid_to_color(i) for i in range(get_num_objects())]
    return torch.tensor(colors)

# Convert tensor to RGB by averaging colors of all objects at each position.
def intid_tensor_to_rgb(data : torch.tensor) -> torch.tensor:
    num_obj = get_num_objects()
    assert data.shape[1] == get_num_objects(), (
        f"Object one-hot tensor got the wrong number of objects ({data.shape[1]}), expected {num_obj}")

    data = data.float()

    # All dimensions after batch and channel dimension are assumed to be spatial.
    num_spatial_dims = len(data.shape) - 2

    rgb_tensor = torch.zeros_like(data[:, :3])
    count_tensor = torch.zeros_like(data[:, :1])
    for c in range(num_obj):
        channel_slice = data[:, c:c+1]
        channel_count_slice = channel_slice > 0.01
        rgb_color = object_intid_to_color(c)

        rgb_color = torch.tensor(rgb_color, device=data.device).unsqueeze(0)
        # Add correct number of spatial dimensions
        for _ in range(num_spatial_dims):
            rgb_color = rgb_color.unsqueeze(2)

        rgb_slice = channel_slice * rgb_color
        count_tensor += channel_count_slice
        rgb_tensor += rgb_slice

    rgb_avg_tensor = rgb_tensor / (count_tensor + 1e-10)
    return rgb_avg_tensor / 255