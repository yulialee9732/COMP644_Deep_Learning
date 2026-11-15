import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt


# Load the four input images from left to right
images = []

for i in range(1, 5):
    img = cv.imread(f"yosemite/yosemite{i}.jpg")
    # img = cv.imread(f"wilson/wilson{i}.jpg")

    if img is None:
        raise ValueError(f"Could not load {i}.jpg")

    # Resize images to reduce computation and memory usage
    img = cv.resize(img, (300, 480))
    images.append(img)


# Step 1: Feature detection and description

def get_features(img):
    # Convert to grayscale because SIFT operates on intensity values
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect SIFT keypoints and compute their descriptors
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)

    return gray, kp, des


# Display detected SIFT keypoints for all four input images
fig, axes = plt.subplots(1, 4, figsize=(14, 5))

for i, img in enumerate(images):
    gray, kp, _ = get_features(img)

    result = cv.drawKeypoints(
        gray,
        kp,
        None,
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    axes[i].imshow(result, cmap="gray")
    axes[i].set_title(f"Image {i + 1}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()
plt.close(fig)


# Step 2: Feature matching and image alignment

def get_matches(des1, des2):
    # Match SIFT descriptors using Euclidean (L2) distance
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # Lower descriptor distance represents a better match
    return sorted(matches, key=lambda x: x.distance)


def get_distance_transform(img):
    # Create a binary mask separating image pixels from the black background
    thresh = cv.threshold(
        img,
        0,
        255,
        cv.THRESH_BINARY
    )[1]

    thresh = thresh.any(axis=2)
    thresh = np.pad(thresh, 1)

    # Compute each pixel's distance from the nearest image boundary
    dist = distance_transform_edt(thresh)[1:-1, 1:-1]
    dist = dist[:, :, None]

    # Normalize the weights
    max_dist = dist.max()

    if max_dist > 0:
        dist = dist / max_dist

    return dist.astype(np.float32)


def stitch_pair(img1, img2, show_details=False):
    # Detect SIFT features in both images
    gray1, kp1, des1 = get_features(img1)
    gray2, kp2, des2 = get_features(img2)

    if des1 is None or des2 is None:
        raise ValueError("Not enough SIFT features")

    # Match descriptors between the two images
    matches = get_matches(des1, des2)

    if len(matches) < 4:
        raise ValueError("Not enough matches")

    # Construct corresponding point arrays from the matched keypoints
    pts1 = np.float32([
        kp1[m.queryIdx].pt for m in matches
    ])

    pts2 = np.float32([
        kp2[m.trainIdx].pt for m in matches
    ])

    # Estimate a robust homography that maps img2 onto img1
    H, mask = cv.findHomography(
        pts2,
        pts1,
        cv.RANSAC
    )

    if H is None:
        raise ValueError("Could not estimate homography")

    print(
        f"Matches: {len(matches)}, "
        f"Inliers: {int(mask.sum())}"
    )

    # Visualize the strongest feature matches for the first image pair
    if show_details:
        match_img = cv.drawMatches(
            gray1,
            kp1,
            gray2,
            kp2,
            matches[:10],
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        fig = plt.figure(figsize=(12, 5))
        plt.imshow(match_img, cmap="gray")
        plt.title("Top 10 Feature Matches")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Define the four corners of each image
    corners1 = np.float32([
        [0, 0],
        [w1, 0],
        [w1, h1],
        [0, h1]
    ]).reshape(-1, 1, 2)

    corners2 = np.float32([
        [0, 0],
        [w2, 0],
        [w2, h2],
        [0, h2]
    ]).reshape(-1, 1, 2)

    # Warp img2's corners to determine the required panorama canvas size
    warped_corners2 = cv.perspectiveTransform(
        corners2,
        H
    )

    all_corners = np.concatenate(
        (corners1, warped_corners2),
        axis=0
    )

    x_min = int(np.floor(all_corners[:, 0, 0].min()))
    y_min = int(np.floor(all_corners[:, 0, 1].min()))
    x_max = int(np.ceil(all_corners[:, 0, 0].max()))
    y_max = int(np.ceil(all_corners[:, 0, 1].max()))

    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        raise ValueError("Invalid panorama size")

    # Translate the panorama so that all coordinates are positive
    tx = -x_min
    ty = -y_min

    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float64)

    # Warp img2 into the coordinate system of img1
    warped_img2 = cv.warpPerspective(
        img2,
        T @ H,
        (width, height)
    )

    # Place img1 onto the same panorama canvas
    img1_canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    img1_canvas[
        ty:ty + h1,
        tx:tx + w1
    ] = img1

    # Show the two original images and their alignment
    if show_details:
        aligned = cv.addWeighted(
            img1_canvas,
            0.5,
            warped_img2,
            0.5,
            0
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
        axes[0].set_title("Image 1")

        axes[1].imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
        axes[1].set_title("Image 2")

        axes[2].imshow(cv.cvtColor(aligned, cv.COLOR_BGR2RGB))
        axes[2].set_title("Aligned")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.show()
        plt.close(fig)

        del aligned


    # Step 3: Blend the aligned images

    # Give pixels farther from image boundaries larger blending weights
    weight1 = get_distance_transform(img1_canvas)
    weight2 = get_distance_transform(warped_img2)

    denom = np.maximum(weight1 + weight2, 1e-6)

    # Compute the weighted average of the two aligned images
    blended = (
        img1_canvas.astype(np.float32) * weight1
        + warped_img2.astype(np.float32) * weight2
    ) / denom

    blended = np.clip(blended, 0, 255).astype(np.uint8)

    if show_details:
        fig = plt.figure(figsize=(12, 6))

        plt.imshow(
            cv.cvtColor(
                blended,
                cv.COLOR_BGR2RGB
            )
        )

        plt.title("Blended Image")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # Delete large temporary arrays before stitching the next image
    del weight1
    del weight2
    del denom
    del img1_canvas
    del warped_img2

    return blended


# Stitch the first two images and display intermediate results
first_panorama = stitch_pair(
    images[0],
    images[1],
    show_details=True
)


# Step 4: Progressively stitch all four images

panorama = first_panorama

# Treat each previous panorama as the first image and add the next image
for i in range(2, len(images)):
    print(f"Adding image {i + 1}...")

    old_panorama = panorama

    panorama = stitch_pair(
        old_panorama,
        images[i]
    )

    del old_panorama


# Display the final four-image panorama
fig = plt.figure(figsize=(14, 6))

plt.imshow(
    cv.cvtColor(
        panorama,
        cv.COLOR_BGR2RGB
    )
)

plt.title("Final 4-Image Panorama")
plt.axis("off")
plt.tight_layout()
plt.show()
plt.close(fig)

print("Finished")