import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def load_image_dataset(
    root_dir: str,
    img_size=(64, 64),
    max_per_class: int = 500,
):
    """
    Minimal image loader.

    - Expects structure: root_dir/class_name/*.png|*.jpg|*.jpeg
    - Converts to grayscale, resizes, flattens to 1D vector.
    - Uses at most `max_per_class` images per class to keep things light.
    """
    root = Path(root_dir)
    class_names = sorted(
        [d.name for d in root.iterdir() if d.is_dir()]
    )[:5]  # at most 5 classes

    X, y = [], []
    for label, cls in enumerate(class_names):
        class_dir = root / cls
        count = 0
        for file_name in class_dir.iterdir():
            if file_name.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            try:
                img = Image.open(file_name).convert("L")  # grayscale
                img = img.resize(img_size)
                arr = np.asarray(img, dtype="float32").reshape(-1) / 255.0
                X.append(arr)
                y.append(label)
                count += 1
                if count >= max_per_class:
                    break
            except Exception:
                # Skip unreadable/broken images
                continue

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=int)
    return X, y, class_names


def run_logistic_regression(X_train, X_test, y_train, y_test, class_names):
    # Scale features and use more iterations for better convergence
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            n_jobs=1,
            random_state=42,
        ),
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("=== Logistic Regression ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )


def run_kmeans(X_train, X_test, y_train, y_test, class_names):
    n_classes = len(class_names)
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    kmeans.fit(X_train)

    # Cluster assignments for test set
    train_clusters = kmeans.labels_
    test_clusters = kmeans.predict(X_test)

    # Map each cluster to the most frequent class in the training data
    cluster_to_label = {}
    for c in range(n_classes):
        mask = train_clusters == c
        if not np.any(mask):
            cluster_to_label[c] = 0
            continue
        labels, counts = np.unique(y_train[mask], return_counts=True)
        cluster_to_label[c] = int(labels[np.argmax(counts)])

    y_pred = np.array([cluster_to_label[c] for c in test_clusters])

    acc = accuracy_score(y_test, y_pred)
    ari = adjusted_rand_score(y_test, test_clusters)

    print("\n=== KMeans (unsupervised) ===")
    print(f"Cluster-label accuracy (via majority mapping): {acc:.4f}")
    print(f"Adjusted Rand Index (clusters vs true labels): {ari:.4f}")
    print("\nClassification-style report (using mapped labels):")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )


def main():
    # Root directory where class folders live
    data_dir = os.path.join(os.path.dirname(__file__), "Data")

    print(f"Loading images from: {data_dir}")
    X, y, class_names = load_image_dataset(
        data_dir,
        img_size=(64, 64),
        max_per_class=2000,  # reduce if memory is an issue
    )
    print(f"Loaded {len(X)} images across {len(class_names)} classes: {class_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    run_logistic_regression(X_train, X_test, y_train, y_test, class_names)
    run_kmeans(X_train, X_test, y_train, y_test, class_names)


if __name__ == "__main__":
    main()


