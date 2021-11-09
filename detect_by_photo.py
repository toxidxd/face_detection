import os
import face_recognition
import shutil


def dataset_create():
    print("Creating dataset...")
    enc_dataset = []
    images = os.listdir("dataset_photo")

    for (i, img) in enumerate(images):
        print(f"+ Processing {i+1}/{len(images)} photo...")
        face_img = face_recognition.load_image_file(f"dataset_photo/{img}")
        face_enc = face_recognition.face_encodings(face_img)

        if len(face_enc) > 0:
            enc_dataset.append(face_enc[0])
        else:
            print("++ No faces on photo!")

    print(f"Dataset crated from {len(enc_dataset)} photos")
    return enc_dataset


def face_rec(dataset):

    if os.path.exists("dataset_photo"):
        print("Directory dataset_photo exists")
    else:
        os.mkdir("dataset_photo")
        print("Directory dataset_photo created")

    photos = os.listdir("photos")
    print(f"Found {len(photos)} photos to recognize\n-------")

    for (i, img) in enumerate(photos):
        print(f"+ Processing {i + 1}/{len(photos)} photo...")
        face_img = face_recognition.load_image_file(f"photos/{img}")
        face_enc = face_recognition.face_encodings(face_img)
        print(f"++ Found {len(face_enc)} face on photo")

        if len(face_enc) > 0:
            for (j, face) in enumerate(face_enc):
                print(f"+++ Processing {j+ 1}/{len(face_enc)} faces...")
                for enc_data in dataset:
                    compare = face_recognition.compare_faces([enc_data], face)
                    if compare[0]:
                        print(f"++++ We have a match in {img}")
                        shutil.copy2(f"photos/{img}", "recognized_photos/")
                        break

        else:
            print("+++ No faces on photo!")


def main():
    dataset = dataset_create()
    face_rec(dataset=dataset)


if __name__ == "__main__":
    main()
