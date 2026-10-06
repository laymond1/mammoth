### compositional templates from: https://github.com/umd-huang-lab/perceptionCLIP ###
from datasets.seq_8vision import Sequential8Vision


def generate_composite_factors(templates, selected_factors=None):
    """
    templates: a dictionary of contextual factors with the following format
    templates = {
        "factor_1": {
            "value_1": ["description_1", "description_2", "description_3"],
            "value_2": ["description_1", "description_2", "description_3"],
            }
        "factor_2": {
            "value_1": ["description_1", "description_2", "description_3"],
            "value_2": ["description_1", "description_2", "description_3"],
            "value_3": ["description_1", "description_2", "description_3"],
            }
        }
    selected_factors: a list of factors that we would like to compose
    """
    # If selected_factors is provided, filter out the keys that are not in the list
    if selected_factors:
        templates = {k: templates[k] for k in selected_factors if k in templates}

    # Base case: if the templates dictionary is empty, return a dictionary with a single empty entry
    if not templates:
        return {"": [""]}

    # Extract the first key-value pair from the templates dictionary
    key, sub_dict = next(iter(templates.items()))

    # Create a copy of the templates dictionary without the extracted key
    rest_templates = {k: v for k, v in templates.items() if k != key}

    composite = {}

    # Iterate over each sub_key and its associated values in the sub_dict
    for sub_key, values in sub_dict.items():
        # Recursively generate composite conditions for the rest of the templates
        for rest_key, rest_values in generate_composite_factors(rest_templates).items():
            # Construct the new composite key
            new_key = f"{sub_key}_{rest_key}" if rest_key else sub_key

            # Combine every value from the current list with every value from the recursive results
            # Add commas during composition, but ensure we don't add unnecessary commas for empty values
            combined_values = [v + (", " + rv if rv else "") if v else rv for v in values for rv in rest_values]
            composite[new_key] = combined_values

    return composite


def compose_template(org_templates, factor_templates):
    factor_templates = [factor_templates[category] for category in factor_templates]
    new_templates = []
    for factors in factor_templates:
        new_factors = []
        for descriptions in factors:
            if descriptions:
                new_factors.append(
                    lambda c, main=org_templates[0], descriptions=descriptions: main(
                        c) + ', ' + descriptions + "."
                )
            else:
                new_factors.append(
                    lambda c, main=org_templates[0], descriptions=descriptions: main(
                        c) + "."
                )

        new_templates.extend(new_factors)

    return new_templates


cars_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'i love my {c}!',
    lambda c: f'a photo of my dirty {c}.',
    lambda c: f'a photo of my clean {c}.',
    lambda c: f'a photo of my new {c}.',
    lambda c: f'a photo of my old {c}.',
]

cifar10_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'a low contrast photo of a {c}.',
    lambda c: f'a high contrast photo of a {c}.',
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a photo of a big {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a low contrast photo of the {c}.',
    lambda c: f'a high contrast photo of the {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the big {c}.',
]

cifar100_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'a low contrast photo of a {c}.',
    lambda c: f'a high contrast photo of a {c}.',
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a photo of a big {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a low contrast photo of the {c}.',
    lambda c: f'a high contrast photo of the {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the big {c}.',
]

dtd_template = [
    lambda c: f'a photo of a {c} texture.',
    lambda c: f'a photo of a {c} pattern.',
    lambda c: f'a photo of a {c} thing.',
    lambda c: f'a photo of a {c} object.',
    lambda c: f'a photo of the {c} texture.',
    lambda c: f'a photo of the {c} pattern.',
    lambda c: f'a photo of the {c} thing.',
    lambda c: f'a photo of the {c} object.',
]

eurosat_template = [
    lambda c: f'a centered satellite photo of {c}.',
    lambda c: f'a centered satellite photo of a {c}.',
    lambda c: f'a centered satellite photo of the {c}.',
]

flowers102_template = [
    lambda c: f'a photo of {c}, a type of flower.',
    lambda c: f'a close-up photo of a {c} in bloom.',
    lambda c: f'a macro photograph of the {c} blossom.',
    lambda c: f'a botanical photo of a {c}.',
    lambda c: f'a photo of the {c} in a garden.',
    lambda c: f'a detailed photo of a {c} flower.',
]

food101_template = [
    lambda c: f'a photo of {c}, a type of food.',
]

gtsrb_template = [
    lambda c: f'a zoomed in photo of a "{c}" traffic sign.',
    lambda c: f'a centered photo of a "{c}" traffic sign.',
    lambda c: f'a close up photo of a "{c}" traffic sign.',
]

oxfordiiitpet_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a close-up portrait of a {c}, a type of pet.',
    lambda c: f'a photo of a {c} looking at the camera.',
    lambda c: f'a {c} sitting indoors.',
    lambda c: f'a {c} posing for a portrait.',
]

pcam_template = [
    lambda c: f'a histopathology image of {c} lymph node tissue.',
    lambda c: f'a microscopic photo showing {c} tissue.',
    lambda c: f'a pathology slide containing {c} cells.',
    lambda c: f'a medical microscopy image of {c} tissue.',
]

mnist_template = [
    lambda c: f'a photo of the number: "{c}".',
]

imagenet_template = [
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a photo of the hard to see {c}.',
    lambda c: f'a low resolution photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a photo of a hard to see {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a photo of a clean {c}.',
    lambda c: f'a photo of a dirty {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a photo of the cool {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a photo of the clean {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a weird {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a photo of the nice {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a photo of a cool {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
]

resisc45_template = [
    lambda c: f'satellite imagery of {c}.',
    lambda c: f'aerial imagery of {c}.',
    lambda c: f'satellite photo of {c}.',
    lambda c: f'aerial photo of {c}.',
    lambda c: f'satellite view of {c}.',
    lambda c: f'aerial view of {c}.',
    lambda c: f'satellite imagery of a {c}.',
    lambda c: f'aerial imagery of a {c}.',
    lambda c: f'satellite photo of a {c}.',
    lambda c: f'aerial photo of a {c}.',
    lambda c: f'satellite view of a {c}.',
    lambda c: f'aerial view of a {c}.',
    lambda c: f'satellite imagery of the {c}.',
    lambda c: f'aerial imagery of the {c}.',
    lambda c: f'satellite photo of the {c}.',
    lambda c: f'aerial photo of the {c}.',
    lambda c: f'satellite view of the {c}.',
    lambda c: f'aerial view of the {c}.',
]

stl10_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
]

sun397_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
]

svhn_template = [
    lambda c: f'a photo of the number: "{c}".',
]

fer2013_template = [
    lambda c: f'a portrait of a person with a {c} expression.',
    lambda c: f'a face showing a {c} emotion.',
    lambda c: f'a close-up of a face that looks {c}.',
    lambda c: f'a person looking {c}.',
    lambda c: f'a facial expression that appears {c}.',
]

cub200_main_template = [
    lambda c: f'a photo of a {c}, a type of bird'
]

cub200_factor_templates = {
    "size": {
        "others": [""],
        "small": ["small"],
        "big": ["big"],
    },
    "background": {
        "others": [""],
        "land": ["on land"],
        "water": ["on water"],
        "forest": ["in forest"],
        "sky": ["in sky"],
        "street": ["on street"],
        "grass": ["on grass"],
        "tree": ["on tree"],
        "flowers": ["with flowers"],
        "beach": ["on beach"],
        "human": ["with human"],
        "branch": ["on a branch"],
    },
    "condition": {
        "normal": [""],
        "cool": ["cool"],
        "nice": ["nice"],
        "weird": ["weird"],
    }
}

cub200_template = compose_template(cub200_main_template, generate_composite_factors(cub200_factor_templates))

dataset_to_template = {
    'Cars': cars_template,
    'CIFAR10': cifar10_template,
    'CIFAR100': cifar100_template,
    'seq-derm7pt': cifar100_template,
    'seq-isic': cifar100_template,
    'seq-cifar100-224-5': cifar100_template,
    'seq-cifar100-224': cifar100_template,
    'seq-cifar100-224-5-permutato': cifar100_template,
    'joint-cifar100': cifar100_template,
    'CUB200': imagenet_template,  # TODO: experiment with the templates for this dataset???
    'CUB200CustomTemplates': cub200_template,
    'DomainNet': imagenet_template,  # TODO: experiment with the templates for this dataset???
    'DTD': dtd_template,
    'seq-dtd': dtd_template,
    'joint-dtd': dtd_template,
    'seq-gtsrb': gtsrb_template,
    'joint-gtsrb': gtsrb_template,
    'seq-mnist': mnist_template,
    'joint-mnist': mnist_template,
    'EuroSAT': eurosat_template,
    'seq-eurosat-rgb': eurosat_template,
    'joint-eurosat-rgb': eurosat_template,
    'seq-resisc45': resisc45_template,
    'joint-resisc45': resisc45_template,
    'seq-cars196': cars_template,
    'joint-cars196': cars_template,
    'seq-cropdisease': cifar100_template,
    'seq-cub200': cub200_main_template,
    'Food101': food101_template,
    'Flowers102': flowers102_template,
    'seq-flowers102': flowers102_template,
    'joint-flowers102': flowers102_template,
    'GTSRB': gtsrb_template,
    'MNIST': mnist_template,
    'seq-mnist-224': mnist_template,
    'joint-mnist-224': mnist_template,
    'ImageNet': imagenet_template,
    'seq-imagenet-r': imagenet_template,
    'seq-imagenet1k': imagenet_template,
    'seq-imagenet21k': imagenet_template,
    'ImageNetR': imagenet_template,
    'OxfordIIITPet': oxfordiiitpet_template,
    'seq-oxfordiiitpet': oxfordiiitpet_template,
    'joint-oxfordiiitpet': oxfordiiitpet_template,
    'PCAM': pcam_template,
    'seq-pcam': pcam_template,
    'joint-pcam': pcam_template,
    'RESISC45': resisc45_template,
    'STL10': stl10_template,
    'seq-stl10': stl10_template,
    'joint-stl10': stl10_template,
    'SUN397': sun397_template,
    'seq-sun397': sun397_template,
    'joint-sun397': sun397_template,
    'SVHN': svhn_template,
    'seq-svhn': svhn_template,
    'joint-svhn': svhn_template,
    'FER2013': fer2013_template,
    'seq-fer2013': fer2013_template,
    'joint-fer2013': fer2013_template,
}

def get_templates(dataset_name):
    if dataset_name.endswith('Val'):
        return get_templates(dataset_name.replace('Val', ''))
    if dataset_name == "seq-8vision":
        return [dataset_to_template[dset_name] for dset_name in Sequential8Vision.DATASET_NAMES]
    assert dataset_name in dataset_to_template, f'Unsupported dataset: {dataset_name}'
    return dataset_to_template[dataset_name]
