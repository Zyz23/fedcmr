import os
import shutil

def format_val():
    val_dir = './data/imagenet_tiny/val'
    img_dir = os.path.join(val_dir, 'images')
    
    if not os.path.exists(img_dir):
        print("✅ 已经整理过了或路径不存在！")
        return

    # 读取标注文件并移动图片
    with open(os.path.join(val_dir, 'val_annotations.txt'), 'r') as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            img_name, class_name = parts[0], parts[1]
            class_dir = os.path.join(val_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            src, dst = os.path.join(img_dir, img_name), os.path.join(class_dir, img_name)
            if os.path.exists(src):
                shutil.move(src, dst)
                
    shutil.rmtree(img_dir)
    print("✅ 验证集格式化完成！")

if __name__ == '__main__':
    format_val()