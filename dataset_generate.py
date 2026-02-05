import io


def process_weibo_data(file_content):
    """
    Processes Weibo data content with updated mapping rules.
    """
    mapping_dict = {
        "000": "0", "001": "0", "010": "0", "100": "0",
        "011": "1", "101": "1", "110": "1", "111": "1"  # 新增110映射
    }
    processed_lines = []

    lines = file_content.strip().splitlines()

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 5:
            try:
                text = parts[1]
                anns = [p.strip() for p in parts[2:5]]  # 取第3-5列作为标注

                # 验证标注有效性
                if all(a in {'0', '1'} for a in anns):
                    key = ''.join(anns)
                    if key in mapping_dict:
                        processed_lines.append(
                            f"{mapping_dict[key]}\t{text}"
                        )
            except (IndexError, ValueError):
                continue  # 跳过格式异常的行

    return processed_lines


# 主程序部分
if __name__ == "__main__":
    final_dataset = []
    output_file = "dataset_my.txt"

    # 定义需要处理的文件路径
    file_paths = [
        ("D:\本科全部东西\大数据原理与技术\大作业\代码\dataset\weibo_media.txt", "weibo_media.txt"),
        ("D:\本科全部东西\大数据原理与技术\大作业\代码\dataset\weibo_supplyment.txt", "weibo_supplyment.txt")
    ]

    for file_path, display_name in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                print(f"成功读取 {display_name}")

                # 处理并合并数据
                final_dataset.extend(process_weibo_data(content))

        except FileNotFoundError:
            print(f"文件 {display_name} 未找到，已跳过")
            continue
        except Exception as e:
            print(f"处理 {display_name} 时出错: {str(e)}")
            continue

    # 写入最终文件
    if final_dataset:
        try:
            with open(output_file, "w", encoding="utf-8") as f_out:
                f_out.write("\n".join(final_dataset))
            print(f"\n处理完成，共生成 {len(final_dataset)} 条数据")
            print(f"结果已保存至: {output_file}")
        except IOError:
            print("写入输出文件失败")
    else:
        print("\n未找到有效数据，未生成输出文件")