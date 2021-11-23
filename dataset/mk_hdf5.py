import numpy
import glob
import h5py
import os


src_path = "/home/ubuntu/data/signate_speed/data"
sequence_path = "/home/ubuntu/data/signate_speed/data/train_videos"

h5_path = "./WFLW.h5"

if __name__ == "__main__":
    with h5py.File(h5_path, "w") as h5:
        distance_image_group = h5.create_group("distance_image")
        for d in sorted(os.listdir(sequence_path)):
            print(d)
            path = os.path.join(sequence_path, d)
            print(path)
            subset_group = distance_image_group.create_group(d)
            exit()

        # jpgファイルを列挙
        for p in sorted(glob.glob(os.path.join(d, "*.jpg"))):
            # numpyで読み込んでDatasetで追加
            image = np.array(Image.open(p)).astype(np.uint8)
            image_dataset = subset_group.create_dataset(
                name=os.path.basename(p), data=image, compression="gzip"
            )
