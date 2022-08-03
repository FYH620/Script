import os
import shutil


def insert_file_names(dir, insert_content, is_front=True):

    # 将该目录下所有文件批量添加文件名前缀或后缀，sample见下方
    # dir[str]: 要处理的目录
    # insert_content[str]: 每个文件名字中要插入的信息
    # is_front[bool]: 前缀插入还是后缀插入

    print("Start Inserting.")
    file_names = os.listdir(dir)
    for step, file_name in enumerate(file_names):

        if is_front:
            new_file_name = insert_content + "_" + file_name
            os.rename(
                os.path.join(dir, file_name),
                os.path.join(dir, new_file_name),
            )

        else:
            fname, ext = os.path.splitext(file_name)
            new_file_name = fname + "_" + insert_content + ext
            os.rename(
                os.path.join(dir, file_name),
                os.path.join(dir, new_file_name),
            )

        if step == 0:
            print(f"For Example:{file_name}->{new_file_name}")

    print("End Inserting.")


def merge_dir_files(input_dir_names, output_dir_name, source_annotations):

    # 将输入的几个目录下的所有文件复制到同一目录下，并对文件来源加入信息，sample见下方
    # input_dir_names[list]: 要复制粘贴文件的所有输入目录
    # output_dir_name[str]: 要粘贴进这些文件的新的目录
    # source_annotations[list]: 在新的目录中指示这些文件来源的标注信息

    if output_dir_name in input_dir_names:
        print(
            f"We merge these input dirs in a new dir.The '{output_dir_name}' is already in '{input_dir_names}'"
        )

    print("Start Merging.")
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)
        print(f"Create a new output dir '{output_dir_name}' done.")

    for step, input_dir_name in enumerate(input_dir_names):
        for file_name in os.listdir(input_dir_name):
            full_file_name = os.path.join(input_dir_name, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, output_dir_name)
                os.rename(
                    os.path.join(output_dir_name, file_name),
                    os.path.join(
                        output_dir_name, source_annotations[step] + "_" + file_name
                    ),
                )

    print("End Merging.")


def test_insert():
    insert_file_names("insert_test_dir", "insert_front", is_front=True)


def test_merge():
    merge_dir_files(
        input_dir_names=["merge_test1_dir", "merge_test2_dir", "merge_test3_dir"],
        output_dir_name="merge_output_dir",
        source_annotations=["source_test1", "source_test2", "source_test3"],
    )


if __name__ == "__main__":
    # test_insert()
    # test_merge()
    pass
