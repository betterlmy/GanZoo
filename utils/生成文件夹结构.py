import os
import subprocess


def generate_markdown_tree(path, ignores=None, output_file=None):
    """
    生成指定路径的文件夹结构的 MD 文件。

    Args:
      path: 指定路径。

    Returns:
      None
    """

    # 当一个函数定义了一个默认参数值时，这个默认值实际上是在函数被定义时创建的，而不是在每次函数调用时创建。
    if ignores is None:
        ignores = ['.git', '.idea', '__pycache__', '.DS_Store']  # 忽略的元素
    if output_file is None:
        output_file = "tree.md"

    result = subprocess.run("/opt/homebrew/bin/tree -i -L 1 --noreport " + path,
                            shell=True, capture_output=True, text=True).stdout.split("\n")

    for i in range(len(result) - 1):

        if i == 0:
            result[i] = "* " + os.path.split(result[i])[-1]
            continue
        ignored = False
        for ignore in ignores:
            if ignore == result[i]:
                ignored = True
                result[i] = ""
                break

        if ignored:
            continue
        # 删除多余的字符
        result[i] = "    * " + result[i]

    with open(os.path.join(path, output_file), "w") as f:
        for res in result:
            if res == "":
                continue
            res = res.replace(r"_", r"\_")
            f.write(res + "\n")
            print(res)


if __name__ == "__main__":
    # path = os.getcwd()
    generate_markdown_tree("/Users/zane/PycharmProjects/GanZoo/cyclegan")
