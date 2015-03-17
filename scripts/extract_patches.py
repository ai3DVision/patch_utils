
import argparse

import numpy as np

import patch_utils

def extract_patches(image, labels, patch_shape):
    """
    Extract all patches where the central pixel has label 1
    and the same amount of patches where the central pixel has
    label 0, randomly sampled.
    """
    # Coordinates where label is 1
    pos_coords = np.argwhere(labels)
    
    # Coordinates where label is 0
    neg_coords = np.argwhere(1 - labels)
    # Sample as many negative coordinates as positive coordinates
    indices = np.arange(len(neg_coords))
    indices = np.random.choice(indices, size=len(pos_coords), replace=False)
    neg_coords = neg_coords[indices]
    
    # Extract patches
    coords = np.vstack([pos_coords, neg_coords])
    patches = patch_utils.get_many_patches(image, patch_shape, coords)
    
    return patches, labels[tuple(coords.T)]

def main():
    
    parser = argparse.ArgumentParser(description="Extract positive and negative 2d patches from a volume")
    parser.add_argument("volume", type=str,
                        help="Image volume")
    parser.add_argument("labels", type=str,
                        help="Label volume")
    parser.add_argument("out_prefix", type=str,
                        help="Prefix of the output files")
    
    args = parser.parse_args()
    
    images = patch_utils.load_volume(args.volume)
    labels = patch_utils.load_volume(args.labels)
    labels = np.uint8(labels == 255)
    
    aux = [extract_patches(image, label, (51,51)) for image, label in zip(images, labels)]
    train_x, train_y = zip(*aux)
    train_x = np.vstack(train_x)
    train_y = np.hstack(train_y)
    
    # Save results
    np.save("%s_x" % args.out_prefix, train_x)
    np.save("%s_y" % args.out_prefix, train_y)

if __name__ == '__main__':
    main()
