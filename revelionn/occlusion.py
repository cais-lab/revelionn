import os
import PIL
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from smqtk_classifier import ClassifyImage
from torchvision import transforms
from xaitk_saliency import GenerateImageClassifierBlackboxSaliency
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.slidingwindow import SlidingWindowStack

from revelionn.mapping_nets.simultaneous_mapping_net import SimultaneousMappingNet
from revelionn.utils.explanation import extract_concepts_from_img

image_filename = ''
class_labels = []
positive_classes_idx = []


def perform_occlusion(main_module, mapping_module, activation_extractor, transformation, img_size,
                      path_to_img, window_size, stride, threads):
    """
    Highlights concepts extracted by the mapping network in the image by occlusion.

    Parameters
    ----------
    main_module : MainModelProcessing
        Class for training, evaluation and processing the main network model.
    mapping_module : MappingModelProcessing
        Class for training, evaluation and processing the mapping network model.
    activation_extractor : ActivationExtractor
        Class for identifying layers of a convolutional neural network and for extracting activations produced during
        network inference.
    transformation : torchvision.transforms
        A transform to apply to the image.
    img_size : int
        The size of the image side.
    path_to_img : str
        Image file path.
    window_size : int
        The block window size.
    stride : int
        The sliding window striding step.
    threads : int
        Optional number threads to use to enable parallelism in applying perturbation masks to an input image.
        If 0, a negative value, or None, work will be performed on the main-thread in-line.

    Returns
    -------
    plt : matplotlib.pyplot
    """

    for transform in transformation.transforms:
        if isinstance(transform, transforms.Normalize):
            mean = transform.mean
            break

    model_loader = transforms.Compose([transforms.ToPILImage(), transformation])
    blackbox_classifier = MultiLabelClassifier(main_module, mapping_module, activation_extractor, model_loader)
    blackbox_fill = np.uint8(np.asarray(mean) * 255)

    gen_slidingwindow = SlidingWindowStack((window_size, window_size), (stride, stride), threads=threads)
    gen_slidingwindow.fill = blackbox_fill

    img = Image.open(path_to_img)
    _, extracted_concepts, _ = extract_concepts_from_img(main_module, mapping_module, img, transformation)

    app(
        path_to_img,
        blackbox_classifier,
        gen_slidingwindow,
        extracted_concepts,
        img_size
    )

    return plt


def app(
        image_filepath: str,
        blackbox_classify: ClassifyImage,
        gen_bb_sal: GenerateImageClassifierBlackboxSaliency,
        extracted_concepts: list[str],
        img_size: int
):
    global image_filename, positive_classes_idx
    image_filename = os.path.split(image_filepath)[-1]
    positive_classes_idx = [i for i, s in enumerate(extracted_concepts) if not s.startswith('Not')]

    ref_image = np.asarray(PIL.Image.open(image_filepath))
    ref_image = cv2.resize(ref_image, (img_size, img_size))
    sal_maps = gen_bb_sal(ref_image, blackbox_classify)
    print(f"Saliency maps: {sal_maps.shape}")
    visualize_saliency(ref_image, sal_maps)


class MultiLabelClassifier(ClassifyImage):

    def __init__(self, main_module, mapping_module, activation_extractor, img_transformation):
        self.main_module = main_module
        self.mapping_module = mapping_module
        self.activation_extractor = activation_extractor
        self.img_transformation = img_transformation
        self.class_labels = self.mapping_module.get_class_labels()
        global class_labels
        class_labels = self.class_labels

    def get_labels(self):
        return [self.class_labels[idx] for idx in positive_classes_idx]

    @torch.no_grad()
    def classify_images(self, image_iter):
        main_net = self.main_module.get_main_net()
        main_net = main_net.eval()

        mapping_net = self.mapping_module.get_mapping_net()
        mapping_net = mapping_net.eval()
        if torch.cuda.is_available():
            main_net = main_net.cuda()
            mapping_net = mapping_net.cuda()

        for img in image_iter:
            image_tensor = self.img_transformation(img).unsqueeze(0)
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            main_output = main_net(image_tensor)
            feature_vec = mapping_net(self.activation_extractor.get_activations(1))

            if isinstance(mapping_net, SimultaneousMappingNet):
                feature_vec = torch.cat(feature_vec, dim=1)
                class_conf = feature_vec.cpu().detach().numpy().squeeze()
                yield dict(zip(self.get_labels(), [class_conf[idx] for idx in positive_classes_idx]))
            else:
                class_conf = feature_vec.cpu().detach().numpy()[0][0]
                yield dict(zip(self.class_labels, [class_conf]))

    def get_config(self):
        # Required by a parent class.
        return {}


def visualize_saliency(ref_image: np.ndarray, sal_maps: np.ndarray) -> None:
    # Visualize the saliency heat-maps
    sub_plot_ind = len(sal_maps) + 1
    plt.figure(figsize=(12, 4), dpi=150)
    plt.subplot(1, sub_plot_ind, 1)
    plt.imshow(ref_image)
    plt.axis('off')
    plt.title(image_filename)

    colorbar_kwargs = {
        "fraction": 0.046 * (ref_image.shape[0] / ref_image.shape[1]),
        "pad": 0.04,
    }

    positive_class_labels = [class_labels[idx] for idx in positive_classes_idx]
    for i, class_sal_map in enumerate(sal_maps):
        print(f"Class {positive_class_labels[i]} saliency map range: [{class_sal_map.min()}, {class_sal_map.max()}]")

        plt.subplot(1, sub_plot_ind, 2 + i)
        plt.imshow(ref_image, alpha=0.8)
        plt.imshow(
            np.clip(class_sal_map, 0, 1),
            cmap='jet', alpha=0.4
        )
        plt.clim(0, 1)
        plt.colorbar(**colorbar_kwargs)
        plt.title(positive_class_labels[i])
        plt.axis('off')

    plt.tight_layout()
