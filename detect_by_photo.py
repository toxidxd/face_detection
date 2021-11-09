# TODO создание датасета по фото и дальнейшее распознавание

"""
1. Создание датасета по фотографиям
2. Перебор фотографий и копирование
3. Сохранить результаты в json
"""
import os
import face_recognition


def dataset_create():
    print("Creating dataset...")
    enc_dataset = []
    images = os.listdir("dataset_photo")

    for (i, img) in enumerate(images):
        print(f"[+] Processing {i+1}/{len(images)} photo...")
        face_img = face_recognition.load_image_file(f"dataset_photo/{img}")
        face_enc = face_recognition.face_encodings(face_img)[0]
        enc_dataset.append(face_enc)

    print(f"Dataset crated from {len(enc_dataset)} photos")
    return enc_dataset


def face_rec(dataset):
    pass


def main():
    dataset = dataset_create()
    face_rec(dataset=dataset)


if __name__ == "__main__":
    main()
