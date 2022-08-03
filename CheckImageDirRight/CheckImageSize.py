import os


def check_file_ext(file_name, check_list):
    _, ext = os.path.splitext(file_name)
    return ext[1:] in check_list


def check_image_disk_size(
    dir,
    size_threshold=1,
    check_list=["bmp", "jpg", "jpeg", "png"],
):

    """
    Args:
        dir(str):            the location of all image files
        size_threshold(int): specify the file size filtering threshold in bytes(default is 1kb)
        check_list(list):    determine whether it is an image suffix list
    """

    print("Start Checking.")

    file_names = os.listdir(dir)
    print(f"Total {len(file_names)} Files In This Dir.")

    for file_name in file_names:
        if check_file_ext(file_name.lower().strip(), check_list):
            file_disk_size = os.stat(os.path.join(dir, file_name)).st_size / 1024
            if file_disk_size <= size_threshold:
                print(
                    f"Please check the file '{file_name}'.Its size suggests it maybe damaged."
                )
        else:
            print(f"Please check the file '{file_name}'.It may not be an image.")

    print("End Checking.")


def test_demo():
    check_image_disk_size("./test_dir")


if __name__ == "__main__":
    test_demo()
